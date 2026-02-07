import streamlit as st
import os
import sys
from pathlib import Path

# Ajout du dossier parent au path pour les imports si nécessaire
current_dir = Path(__file__).parent
sys.path.append(str(current_dir))

from query_rag import RAGAgent

# Configuration de la page
st.set_page_config(
    page_title="Mairie de Triffouillis - Assistant RAG",
    page_icon="🤖",
    layout="centered"
)

# Style CSS personnalisé
st.markdown("""
    <style>
    .main {
        background-color: #f5f7f9;
    }
    .stChatMessage {
        border-radius: 15px;
        padding: 10px;
        margin-bottom: 10px;
    }
    .source-box {
        background-color: #e1e8ef;
        padding: 10px;
        border-radius: 5px;
        font-size: 0.8rem;
        margin-top: 5px;
    }
    </style>
    """, unsafe_allow_html=True)

@st.cache_resource
def load_agent():
    """Charge l'agent une seule fois et le garde en mémoire."""
    try:
        return RAGAgent()
    except Exception as e:
        st.error(f"Erreur d'initialisation de l'agent : {e}")
        return None

# Initialisation de l'agent
agent = load_agent()

# --- Sidebar pour les paramètres ---
with st.sidebar:
    st.title("⚙️ Paramètres RAG")
    
    st.subheader("Récupération (Retrieval)")
    top_k = st.slider("Nombre de documents (Top-K)", min_value=1, max_value=10, value=3)
    
    st.subheader("Génération (LLM)")
    temp = st.slider("Température (Créativité)", min_value=0.0, max_value=1.0, value=0.2, step=0.1)
    top_p = st.slider("Top-P (Diversité)", min_value=0.0, max_value=1.0, value=1.0, step=0.1)

    if st.button("Effacer l'historique"):
        st.session_state.messages = []
        st.rerun()
    
    st.divider()
    st.write("### À propos")
    st.write("Cet assistant utilise un système RAG (Retrieval Augmented Generation) basé sur :")
    st.write("- **Embeddings :** Google Gemini")
    st.write("- **LLM :** Kimi K2 (Groq)")

# --- Contenu Principal ---
st.title("🤖 Assistant de la Mairie")
st.subheader("Triffouillis-sur-Loire")
st.info("Je réponds à vos questions sur les services municipaux, les travaux et les événements.")

# Initialisation de l'historique dans la session
if "messages" not in st.session_state:
    st.session_state.messages = []

# Affichage des messages passés
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if "docs" in message and message["docs"]:
            with st.expander("Sources consultées"):
                for i, doc in enumerate(message["docs"]):
                    source_name = doc.get('source') or doc.get('document_source') or "Inconnue"
                    st.markdown(f"**Source {i+1} :** {source_name}")
                    st.caption(doc.get('content', '')[:200] + "...")

# Zone de saisie utilisateur
if prompt := st.chat_input("Comment puis-je vous aider ?"):
    # Affichage du message utilisateur
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Génération de la réponse
    with st.chat_message("assistant"):
        if agent:
            with st.spinner("Recherche et génération..."):
                try:
                    # On passe l'historique pour la reformulation
                    history = [{"role": m["role"], "content": m["content"]} for m in st.session_state.messages[:-1]]
                    
                    # Appel avec les paramètres dynamiques
                    answer, docs = agent.ask(
                        prompt, 
                        history=history, 
                        k=top_k, 
                        temperature=temp, 
                        top_p=top_p
                    )
                    
                    st.markdown(answer)
                    
                    # Affichage des sources
                    if docs:
                        with st.expander("Sources consultées"):
                            for i, doc in enumerate(docs):
                                source_name = doc.get('source') or doc.get('document_source') or "Inconnue"
                                st.markdown(f"**Source {i+1} :** {source_name}")
                                st.caption(doc.get('content', '')[:300] + "...")
                    
                    # Sauvegarde dans la session
                    st.session_state.messages.append({
                        "role": "assistant", 
                        "content": answer,
                        "docs": docs
                    })
                except Exception as e:
                    error_msg = f"Désolé, une erreur est survenue : {str(e)}"
                    st.error(error_msg)
        else:
            st.error("L'agent n'a pas pu être chargé. Vérifiez vos clés API.")