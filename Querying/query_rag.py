import os
import json
import logging
import numpy as np
import faiss
import google.generativeai as genai
import time
from groq import Groq
from dotenv import load_dotenv
from typing import List, Dict, Tuple, Optional

# Configuration du logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

load_dotenv()

# ==============================================================================
# CONFIGURATION
# ==============================================================================
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, '..'))

INDEX_DIR = os.path.join(BASE_DIR, 'faiss_indexes', 'gemini')
INDEX_FILE = os.path.join(INDEX_DIR, 'index.bin')
METADATA_FILE = os.path.join(INDEX_DIR, 'metadata.json')

EMBEDDING_MODEL = "models/gemini-embedding-001"
GENERATION_MODEL = "moonshotai/kimi-k2-instruct-0905"

# Templates de Prompts
SYSTEM_PROMPT = """
RÔLE : Assistant virtuel de la Mairie de Triffouillis-sur-Loire.

DIRECTIVES :
1. Si la question est une salutation (ex: bonjour, salut) ou un remerciement, réponds poliment sans chercher dans les documents.
2. Pour les questions d'information, utilise UNIQUEMENT le CONTEXTE DOCUMENTAIRE fourni ci-dessous.
3. Si l'information demandée n'est pas dans le contexte, réponds : "Je ne dispose pas de cette information, veuillez contacter le secrétariat."
4. Ton : Professionnel, chaleureux et courtois.

HISTORIQUE : {history}
CONTEXTE : {context}
QUESTION : {query}
"""

INTENT_PROMPT = """Classifie l'intention : SEARCH (demande d'info) ou CHAT (salutation/merci). Phrase : "{query}". Réponds par un seul mot."""

REFORMULATION_PROMPT = """Reformule la question pour qu'elle soit autonome en utilisant l'historique : {history}. Question : {query}."""

class RAGAgent:
    def __init__(self, google_key: str = None, groq_key: str = None):
        self.google_key = google_key or os.environ.get("GOOGLE_API_KEY")
        self.groq_key = groq_key or os.environ.get("GROQ_API_KEY")
        
        if not self.google_key or not self.groq_key:
            raise ValueError("Clés API manquantes (GOOGLE_API_KEY ou GROQ_API_KEY).")

        genai.configure(api_key=self.google_key)
        self.groq = Groq(api_key=self.groq_key)
        
        self.index, self.metadata = self._load_resources()

    def _load_resources(self):
        if not os.path.exists(INDEX_FILE):
            logger.warning(f"Index non trouvé à {INDEX_FILE}")
            return None, None
        
        try:
            index = faiss.read_index(INDEX_FILE)
            with open(METADATA_FILE, 'r', encoding='utf-8') as f:
                metadata = json.load(f)
            return index, metadata
        except Exception as e:
            logger.error(f"Erreur chargement ressources : {e}")
            return None, None

    def _generate(self, prompt: str, temp=0.2, top_p=1.0) -> str:
        try:
            chat_completion = self.groq.chat.completions.create(
                model=GENERATION_MODEL,
                messages=[{"role": "user", "content": prompt}],
                temperature=temp,
                top_p=top_p,
                max_tokens=2048
            )
            return chat_completion.choices[0].message.content.strip()
        except Exception as e:
            logger.error(f"Erreur Groq : {e}")
            return "Désolé, une erreur technique est survenue."

    def ask(self, query: str, history: List[Dict] = None, k=3, temperature=0.2, top_p=1.0) -> Tuple[str, List[Dict]]:
        history = history or []
        history_text = "\n".join([f"{m['role']}: {m['content']}" for m in history[-5:]])

        # 1. Classification & Reformulation
        intent = self._generate(INTENT_PROMPT.format(query=query), temp=0)
        
        search_query = query
        if history:
            search_query = self._generate(REFORMULATION_PROMPT.format(history=history_text, query=query), temp=0.1)

        # 2. Recherche si nécessaire
        docs = []
        if "SEARCH" in intent.upper() and self.index:
            max_retries = 3
            wait_time = 5
            for attempt in range(max_retries):
                try:
                    emb = genai.embed_content(model=EMBEDDING_MODEL, content=search_query, task_type="retrieval_query")
                    vec = np.array([emb['embedding']]).astype('float32')
                    dist, indices = self.index.search(vec, k)
                    
                    for i, idx in enumerate(indices[0]):
                        if idx != -1:
                            docs.append(self.metadata[idx])
                    break
                except Exception as e:
                    if "429" in str(e) and attempt < max_retries - 1:
                        logger.warning(f"Quota atteint (429). Attente de {wait_time}s...")
                        time.sleep(wait_time)
                        wait_time *= 2
                    else:
                        logger.error(f"Erreur recherche : {e}")
                        break

        # 3. Génération
        if "CHAT" in intent.upper():
            context_text = "L'utilisateur engage une conversation polie. Pas de recherche documentaire nécessaire."
        elif docs:
            context_text = ""
            for d in docs:
                source_name = d.get('source') or d.get('document_source') or "Inconnue"
                context_text += f"\nSource {source_name}: {d.get('content', '')}"
        else:
            context_text = "Aucun document trouvé dans la base de données."
        
        final_prompt = SYSTEM_PROMPT.format(history=history_text, context=context_text, query=query)
        
        answer = self._generate(final_prompt, temp=temperature, top_p=top_p)
        return answer, docs


# ==============================================================================
# BOUCLE PRINCIPALE (CLI)
# ==============================================================================

def main():
    try:
        # L'agent charge automatiquement GOOGLE_API_KEY et GROQ_API_KEY via .env
        agent = RAGAgent()
    except Exception as e:
        print(f"❌ Erreur d'initialisation : {e}")
        print("Vérifiez que votre fichier .env contient GOOGLE_API_KEY et GROQ_API_KEY.")
        return

    print("\n" + "="*60)
    print("🤖 AGENT D'ACCUEIL MAIRIE (KIMI K2 + GEMINI EMBEDDINGS)")
    print("   -> Mode : Questions/Réponses interactif (CLI)")
    print("   -> Tapez 'exit', 'quit' ou 'q' pour quitter.")
    print("="*60)
    
    cli_history = []

    while True:
        query = input("\n❓ Votre question : ")
        if query.lower() in ['exit', 'quit', 'q']:
            print("Au revoir !")
            break
        
        print("   🔍 Analyse et génération...")
        try:
            answer, docs = agent.ask(query, history=cli_history)
            
            cli_history.append({"role": "user", "content": query})
            cli_history.append({"role": "assistant", "content": answer})
            
            print("\n" + "-"*60)
            print(answer)
            print("-"*60)
            if docs:
                print(f"   (Basé sur {len(docs)} documents sources)")
                for i, d in enumerate(docs):
                    s = d.get('source') or d.get('document_source') or "Inconnu"
                    print(f"     - {s}")
        except Exception as e:
            print(f"❌ Erreur : {e}")

if __name__ == "__main__":
    main()