
import os
import glob
import numpy as np
import requests
import re
import json
import logging
from typing import List, Dict, Any

from sklearn.manifold import TSNE
from sklearn.metrics.pairwise import cosine_similarity
import plotly.graph_objects as go

# --- Modèles d'embedding ---
from sentence_transformers import SentenceTransformer
from gensim.models.fasttext import load_facebook_model
# Nécessite : pip install google-generativeai
import google.generativeai as genai 

# ==============================================================================
# CONFIGURATION
# ==============================================================================

# --- Configuration des chemins relative au script ---
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, '..'))

# 1. Chemin vers les chunks JSON générés par chunker.py
CHUNKED_DOCS_DIR = os.path.join(BASE_DIR, 'chunked_documents_by_tags') 

# 2. Nombre de chunks à analyser (pour la démo t-SNE)
MAX_CHUNKS_TO_ANALYZE = 300 

# 3. Configuration des modèles
SBERT_MODEL_NAME = 'sentence-transformers/all-MiniLM-L6-v2'
GEMINI_EMBED_MODEL = "models/text-embedding-004"

# 4. Chemin et URL pour le modèle FastText (français)
FASTTEXT_MODEL_PATH = 'cc.fr.300.bin'
FASTTEXT_DOWNLOAD_URL = "https://dl.fbaipublicfiles.com/fasttext/vectors-wiki/cc.fr.300.vec.gz" 

# ==============================================================================
# FONCTIONS AUXILIAIRES
# ==============================================================================

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

def clean_text(text: str) -> str:
    """Nettoie le texte."""
    text = text.replace('\n', ' ').replace('\r', ' ')
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

def load_chunks(directory: str, max_chunks: int) -> tuple[List[str], List[str]]:
    """Charge le contenu et les labels des chunks à partir des fichiers JSON."""
    filepaths = glob.glob(os.path.join(directory, '*.json'))
    all_texts = []
    all_labels = []

    for filepath in filepaths:
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                data = json.load(f)
                filename_base = os.path.basename(filepath).replace('.json', '')

                for i, chunk in enumerate(data):
                    if len(all_texts) >= max_chunks: break
                    
                    text = clean_text(chunk.get('content', ''))
                    if text:
                        all_texts.append(text)
                        
                        metadata = chunk.get('metadata', {})
                        label = f"{filename_base} (C{i})"
                        
                        # Afficher le titre sémantique si disponible (meilleur label)
                        if metadata.get('titre_complet'):
                            label = f"{filename_base[:15]} | {metadata['titre_complet'][:20]}..."
                            
                        all_labels.append(label)
                
        except Exception as e:
            logging.error(f"Erreur lors du chargement ou du parsing du fichier JSON {filepath}: {e}")
            continue

    logging.info(f"{len(all_texts)} chunks chargés et prêts pour l'embedding (max: {max_chunks}).")
    return all_texts, all_labels

def visualize_embeddings(embeddings, labels, title='Visualisation des Embeddings'):
    """Réduit la dimension des embeddings avec t-SNE et les affiche avec Plotly."""
    if len(embeddings) < 5:
        logging.warning("Pas assez de données pour la visualisation t-SNE (min 5).")
        return
        
    logging.info(f"Génération de la visualisation pour '{title}'...")
    
    # Perplexity doit être ajustée à la taille du jeu de données
    perplexity_val = min(50, len(embeddings) - 1)
    
    tsne = TSNE(n_components=2, perplexity=perplexity_val, random_state=42, init='pca', learning_rate=200)
    embeddings_2d = tsne.fit_transform(embeddings)

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=embeddings_2d[:, 0],
        y=embeddings_2d[:, 1],
        mode='markers',
        text=labels,
        hoverinfo='text',
        marker=dict(size=8, opacity=0.8)
    ))
    
    fig.update_layout(title=title, showlegend=False)
    output_filename = f"visualisation_{title.replace(' ', '_').lower()}.html"
    fig.write_html(output_filename)
    logging.info(f"Visualisation sauvegardée dans '{output_filename}'")


# ==============================================================================
# LOGIQUE PRINCIPALE D'ANALYSE
# ==============================================================================

def main_embedder():
    
    # 1. Chargement des données (CHUNKS)
    texts, labels = load_chunks(CHUNKED_DOCS_DIR, MAX_CHUNKS_TO_ANALYZE)
    
    if not texts:
        logging.info("Aucun chunk à traiter. Arrêt.")
        return

    # --- 2. Analyse avec Gemini ---
    logging.info("\n" + "="*50)
    logging.info("DÉBUT DE L'ANALYSE : GEMINI")
    
    api_key = os.environ.get("GOOGLE_API_KEY") 
    
    if api_key:
        try:
            genai.configure(api_key=api_key)
            all_embeddings = []
            batch_size = 100  # Gemini a souvent une limite de batch
            
            for i in range(0, len(texts), batch_size):
                batch = texts[i:i+batch_size]
                response = genai.embed_content(
                    model=GEMINI_EMBED_MODEL,
                    content=batch,
                    task_type="retrieval_document"
                )
                all_embeddings.extend(response['embedding'])

            gemini_embeddings = np.array(all_embeddings)
            visualize_embeddings(gemini_embeddings, labels, title="Embeddings Gemini")

            similarity_matrix = cosine_similarity(gemini_embeddings)
            logging.info("\nMatrice de similarité (Gemini, 5x5) :")
            print(np.round(similarity_matrix[:5, :5], 2)) 

        except Exception as e:
            logging.error(f"Erreur lors de la génération des embeddings Gemini : {e}")
    else:
        logging.warning("Variable d'environnement GOOGLE_API_KEY non trouvée. L'analyse Gemini est ignorée.")

    # --- 3. Analyse avec Sentence-BERT ---
    logging.info("\n" + "="*50)
    logging.info("DÉBUT DE L'ANALYSE : SENTENCE-BERT")
    
    try:
        sbert_model = SentenceTransformer(SBERT_MODEL_NAME)
        sbert_embeddings = sbert_model.encode(texts, show_progress_bar=True)
        visualize_embeddings(sbert_embeddings, labels, title="Embeddings Sentence-BERT")
        
        # AJOUT : Calcul et affichage de la Matrice de Similarité SBERT
        similarity_matrix_sbert = cosine_similarity(sbert_embeddings)
        logging.info("\nMatrice de similarité (Sentence-BERT, 5x5) :")
        print(np.round(similarity_matrix_sbert[:5, :5], 2)) 
        
    except Exception as e:
        logging.error(f"Erreur lors de la génération des embeddings Sentence-BERT : {e}")


    # --- 4. Analyse avec FastText ---
    logging.info("\n" + "="*50)
    logging.info("DÉBUT DE L'ANALYSE : FASTTEXT")
    
# --- 4. Analyse avec FastText ---
    # ... (code de vérification/chargement du modèle) ...
    try:
        if os.path.exists(FASTTEXT_MODEL_PATH):
            ft_model = load_facebook_model(FASTTEXT_MODEL_PATH)
            fasttext_embeddings = np.array([ft_model.get_sentence_vector(text) for text in texts])
            visualize_embeddings(fasttext_embeddings, labels, title="Embeddings FastText")
            
            # AJOUT : Calcul et affichage de la Matrice de Similarité FastText
            similarity_matrix_ft = cosine_similarity(fasttext_embeddings)
            logging.info("\nMatrice de similarité (FastText, 5x5) :")
            print(np.round(similarity_matrix_ft[:5, :5], 2))

        else:
            logging.warning("Le modèle FastText n'a pas pu être chargé. L'analyse est ignorée.")
    except Exception as e:
        logging.error(f"Erreur lors de la génération des embeddings FastText : {e}")
        
if __name__ == "__main__":
    main_embedder()