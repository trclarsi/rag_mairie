import streamlit as st
import os
import sys
import time
import shutil

# Ajout du dossier parent au path pour importer query_rag et pipeline
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(SCRIPT_DIR)

from query_rag import RAGAgent
from pipeline import run_full_pipeline

# Configuration des chemins pour l'upload
CORPUS_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, '..', 'Corpus'))

# ==============================================================================
# CONFIGURATION DE LA PAGE
# ==============================================================================
st.set_page_config(
    page_title="Mairie Assistant",
    page_icon="🏛️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ... (Le reste du CSS reste inchangé) ...
st.markdown("""
<style>
    .stChatMessage { border-radius: 10px; padding: 10px; }
    .stButton button { border-radius: 20px; }
    .source-box { background-color: #f8f9fa; padding: 15px; border-radius: 8px; border-left: 4px solid #007bff; margin-bottom: 10px; font-size: 0.95em; line-height: 1.4; color: #333; box-shadow: 0 1px 3px rgba(0,0,0,0.1); }
    .source-title { font-weight: bold; color: #007bff; margin-bottom: 5px; display: block; }
    .similarity-score { background-color: #e9ecef; padding: 2px 6px; border-radius: 4px; font-size: 0.8em; font-weight: bold; color: #495057; margin-left: 10px; }
</style>
""", unsafe_allow_html=True)

# ==============================================================================
# BARRE LATÉRALE (SIDEBAR)
# ==============================================================================
with st.sidebar:
    st.image("https://img.icons8.com/color/96/city-hall.png", width=80)
    st.title("Paramètres")
    
    # 1. Initialisation Automatique de l'Agent
    if "agent" not in st.session_state:
        try:
            with st.spinner("Initialisation..."):
                st.session_state.agent = RAGAgent()
            st.success("✅ Assistant prêt")
        except Exception as e:
            st.error("⚠️ Assistant non initialisé")
            st.caption("Ajoutez des documents et lancez l'indexation.")
    else:
        st.success("✅ Assistant Connecté")

    st.divider()

    # 2. Gestion des documents
    st.subheader("📂 Base de connaissance")
    uploaded_files = st.file_uploader("Ajouter des documents", accept_multiple_files=True)
    
    if uploaded_files:
        if st.button("📥 Enregistrer", use_container_width=True):
            if not os.path.exists(CORPUS_DIR): os.makedirs(CORPUS_DIR)
            for f in uploaded_files:
                with open(os.path.join(CORPUS_DIR, f.name), "wb") as out:
                    shutil.copyfileobj(f, out)
            st.toast(f"{len(uploaded_files)} fichiers ajoutés !")
    
    if st.button("🚀 Mettre à jour l'index", use_container_width=True):
        with st.status("Traitement du corpus...", expanded=True) as status:
            try:
                st.write("Exécution de la pipeline (Docling + Whisper)...")
                if run_full_pipeline():
                    st.write("Rechargement de la mémoire...")
                    st.session_state.agent = RAGAgent()
                    status.update(label="Index mis à jour !", state="complete")
                    st.rerun()
                else:
                    status.update(label="Erreur d'indexation", state="error")
            except Exception as e:
                st.error(f"Erreur pipeline : {e}")
                status.update(label="Échec de la pipeline", state="error")

    st.divider()

    # 3. Paramètres Avancés
    st.subheader("⚙️ Réglages")
    temperature = st.slider("Créativité (Température)", 0.0, 1.0, 0.2, 0.1)
    top_p = st.slider("Top-P", 0.0, 1.0, 0.9, 0.05)
    top_k = st.slider("Nombre de documents (Top-K)", 1, 10, 3, 1)
    debug_mode = st.toggle("Mode Expert (Debug)", value=False)

    st.divider()
    
    if st.button("🗑️ Effacer la discussion", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

    st.markdown("---")
    st.markdown("Version 3.1 • *Kimi K2 & Pipeline Intégrée*")

# ==============================================================================
# ZONE PRINCIPALE (Le reste reste identique)
# ==============================================================================
st.title("🏛️ Mairie de Triffouillis-sur-Loire")
st.markdown("##### *Votre assistant virtuel pour toutes les démarches municipales.*")

# Initialisation de l'historique
if "messages" not in st.session_state:
    st.session_state.messages = []

# --- Affichage de l'historique ---
for message in st.session_state.messages:
    role = message["role"]
    avatar = "🧑‍💻" if role == "user" else "🏛️"
    with st.chat_message(role, avatar=avatar):
        st.markdown(message["content"])
        if "sources" in message and message["sources"]:
            with st.expander(f"📚 {len(message['sources'])} Documents consultés"):
                for idx, doc in enumerate(message["sources"]):
                    st.markdown(f"""<div class="source-box"><span class="source-title">Source {idx+1} : {doc.get('source', 'Inconnu')}</span><div>{doc['content'][:400]}...</div></div>""", unsafe_allow_html=True)

# --- Zone de saisie ---
if "agent" in st.session_state:
    if prompt := st.chat_input("Posez votre question ici..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user", avatar="🧑‍💻"):
            st.markdown(prompt)

        with st.chat_message("assistant", avatar="🏛️"):
            message_placeholder = st.empty()
            with st.spinner("Recherche..."):
                try:
                    answer, sources = st.session_state.agent.ask(prompt, history=st.session_state.messages, k=top_k, temperature=temperature, top_p=top_p)
                    message_placeholder.markdown(answer)
                    if sources:
                        with st.expander(f"📚 Voir les sources"):
                            for idx, doc in enumerate(sources):
                                st.markdown(f"""<div class="source-box"><span class="source-title">Source {idx+1} : {doc.get('source', 'Inconnu')}</span><div>{doc['content'][:400]}...</div></div>""", unsafe_allow_html=True)
                    st.session_state.messages.append({"role": "assistant", "content": answer, "sources": sources})
                except Exception as e:
                    st.error(f"Erreur : {e}")
else:
    st.warning("⚠️ L'agent n'est pas initialisé. Ajoutez des documents et lancez l'indexation si nécessaire.")