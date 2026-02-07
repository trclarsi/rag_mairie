import os
import json
import glob
import numpy as np
import faiss
import argparse
import sys
from typing import List, Dict, Any

# --- IMPORT DES LIBRAIRIES DE MODÈLES ---
from sentence_transformers import SentenceTransformer
from gensim.models.fasttext import load_facebook_model
# pip install google-generativeai
import google.generativeai as genai

# ==============================================================================
# CONFIGURATION DES CHEMINS
# ==============================================================================

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, '..'))

# Entrée : Dossier des chunks (Généré par chunker.py)
CHUNKED_DOCS_DIR = os.path.join(BASE_DIR, 'chunked_documents_by_tags')

# Sortie : Dossier où l'index FAISS sera stocké
FAISS_OUTPUT_DIR = os.path.join(SCRIPT_DIR, 'faiss_indexes')

# Chemins spécifiques aux modèles locaux
FASTTEXT_MODEL_PATH = os.path.join(BASE_DIR, 'embedding', 'cc.fr.300.bin')

# ==============================================================================
# FONCTIONS UTILITAIRES
# ==============================================================================

def ask_user_for_method() -> str:
    """Demande à l'utilisateur de choisir le modèle de manière interactive."""
    print("\n" + "="*60)
    print("🤖 CHOIX DU MODÈLE D'EMBEDDING POUR L'INDEXATION")
    print("="*60)
    print("Basé sur votre analyse précédente (embedder.py), quel modèle souhaitez-vous utiliser ?")
    print("\n1. Sentence-BERT (sbert)")
    print("   -> Rapide, local, gratuit. Recommandé pour commencer.")
    print("\n2. Gemini (gemini)")
    print("   -> Très performant, mais nécessite une CLÉ API (Google).")
    print("\n3. FastText (fasttext)")
    print("   -> Statistique, local. Utile si SBERT est trop lent.")
    
    while True:
        choice = input("\n👉 Entrez le numéro (1, 2, 3) ou le nom : ").lower().strip()
        
        if choice in ['1', 'sbert']:
            return 'sbert'
        elif choice in ['2', 'gemini']:
            return 'gemini'
        elif choice in ['3', 'fasttext']:
            return 'fasttext'
        else:
            print("❌ Choix invalide. Veuillez taper 1, 2 ou 3.")

def get_embeddings(method: str, texts: List[str]) -> np.ndarray:
    """Génère les embeddings selon la méthode choisie."""
    print(f"\n🚀 Démarrage de la vectorisation avec : {method.upper()}")
    
    embeddings = []

    # --- CAS 1 : SENTENCE-BERT ---
    if method == 'sbert':
        model_name = 'sentence-transformers/all-MiniLM-L6-v2'
        print(f"Chargement du modèle : {model_name}")
        model = SentenceTransformer(model_name)
        embeddings = model.encode(texts, show_progress_bar=True, convert_to_numpy=True)

    # --- CAS 2 : GEMINI ---
    elif method == 'gemini':
        api_key = os.environ.get("GOOGLE_API_KEY")
        if not api_key:
            print("⚠️ Variable d'environnement GOOGLE_API_KEY introuvable.")
            api_key = input("Veuillez coller votre clé API Google Gemini ici : ").strip()
            if not api_key:
                raise ValueError("Clé API manquante.")
        
        genai.configure(api_key=api_key)
        model_name = "models/text-embedding-004"
        print(f"Appel API Gemini ({model_name})...")
        
        try:
            batch_size = 100
            for i in range(0, len(texts), batch_size):
                batch = texts[i:i+batch_size]
                # Le task_type est important pour l'indexation
                resp = genai.embed_content(
                    model=model_name,
                    content=batch,
                    task_type="retrieval_document"
                )
                embeddings.extend(resp['embedding'])
            
            embeddings = np.array(embeddings)
            
        except Exception as e:
            print(f"❌ Erreur API Gemini : {e}")
            sys.exit(1)

    # --- CAS 3 : FASTTEXT ---
    elif method == 'fasttext':
        if not os.path.exists(FASTTEXT_MODEL_PATH):
            raise FileNotFoundError(f"Modèle FastText introuvable ici : {FASTTEXT_MODEL_PATH}")
        
        print("Chargement du modèle FastText...")
        ft_model = load_facebook_model(FASTTEXT_MODEL_PATH)
        print("Vectorisation...")
        embeddings = np.array([ft_model.get_sentence_vector(t) for t in texts])

    else:
        raise ValueError(f"Méthode inconnue : {method}")

    return embeddings.astype('float32')

# ==============================================================================
# LOGIQUE PRINCIPALE
# ==============================================================================

def main():
    # 1. Gestion des Arguments (Optionnel maintenant)
    parser = argparse.ArgumentParser()
    parser.add_argument('--method', type=str, choices=['sbert', 'gemini', 'fasttext'],
                        help="Méthode d'embedding (optionnel, sinon demande interactive)")
    args = parser.parse_args()

    # SI l'argument n'est pas donné, on lance le menu interactif
    method_chosen = args.method
    if not method_chosen:
        method_chosen = ask_user_for_method()

    # 2. Préparation des dossiers
    if not os.path.exists(FAISS_OUTPUT_DIR):
        os.makedirs(FAISS_OUTPUT_DIR)
    
    method_output_dir = os.path.join(FAISS_OUTPUT_DIR, method_chosen)
    if not os.path.exists(method_output_dir):
        os.makedirs(method_output_dir)

    # 3. Chargement des Chunks JSON
    print(f"\n📂 Lecture des données depuis : {CHUNKED_DOCS_DIR}")
    json_files = glob.glob(os.path.join(CHUNKED_DOCS_DIR, '*.json'))
    
    if not json_files:
        print("❌ Aucun fichier JSON trouvé. Avez-vous lancé chunker.py ?")
        return

    all_texts = []
    metadata_mapping = [] 
    current_id = 0

    for filepath in json_files:
        with open(filepath, 'r', encoding='utf-8') as f:
            try:
                chunks = json.load(f)
                for chunk in chunks:
                    text_content = chunk.get('content', '').strip()
                    if text_content:
                        all_texts.append(text_content)
                        metadata_mapping.append({
                            "faiss_id": current_id,
                            "document_source": chunk.get('document_id'),
                            "content": text_content,
                            "metadata": chunk.get('metadata')
                        })
                        current_id += 1
            except json.JSONDecodeError:
                print(f"⚠️ Fichier corrompu ignoré : {filepath}")

    print(f"✅ {len(all_texts)} chunks chargés.")

    # 4. Génération des Embeddings
    vectors = get_embeddings(method_chosen, all_texts)
    
    # 5. Création de l'Index FAISS
    dimension = vectors.shape[1]
    print(f"\n⚙️ Création de l'index FAISS (Dim={dimension})...")
    
    index = faiss.IndexFlatL2(dimension)
    index.add(vectors)
    
    # 6. Sauvegardes
    index_filename = "index.bin"
    mapping_filename = "metadata.json"
    
    faiss.write_index(index, os.path.join(method_output_dir, index_filename))
    
    with open(os.path.join(method_output_dir, mapping_filename), 'w', encoding='utf-8') as f:
        json.dump(metadata_mapping, f, indent=4, ensure_ascii=False)

    print("\n" + "="*60)
    print(f"🎉 INDEXATION TERMINÉE AVEC SUCCÈS !")
    print(f"🔹 Méthode utilisée : {method_chosen.upper()}")
    print(f"🔹 Index FAISS      : {os.path.join(method_output_dir, index_filename)}")
    print(f"🔹 Métadonnées      : {os.path.join(method_output_dir, mapping_filename)}")
    print("="*60)

if __name__ == "__main__":
    main()