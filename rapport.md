# Rapport d'Analyse du Projet RAG - Mairie de Triffouillis-sur-Loire

Ce document présente une analyse détaillée du système de **RAG (Retrieval-Augmented Generation)** développé pour l'assistant virtuel de la mairie de Triffouillis-sur-Loire.

## 1. Vue d'ensemble

Le projet vise à fournir une interface conversationnelle capable de répondre aux questions des citoyens en se basant sur un corpus documentaire municipal (budgets, bulletins, règlements, compte-rendus, etc.).

**Architecture Globale :**
Le système suit un pipeline RAG classique :
`Ingestion -> Traitement (Markdown) -> Découpage (Chunking) -> Vectorisation (Embedding) -> Indexation (FAISS) -> Recherche & Génération (LLM)`

## 2. Gestion de l'Environnement et Dépendances

Le projet utilise **Poetry** pour la gestion des dépendances et de l'environnement virtuel, garantissant la reproductibilité.

*   **Fichier de configuration :** `pyproject.toml`
*   **Version Python :** `>=3.11`
*   **Dépendances clés :**
    *   **IA & NLP :** `langchain-google-genai`, `google-generativeai` (Gemini), `mistralai`, `sentence-transformers` (via `genai` ou local), `gensim` (FastText), `openai-whisper` (Audio), `ragas` (Évaluation).
    *   **Vector Store :** `faiss-cpu`.
    *   **Traitement de documents :** `docling`.
    *   **Data Science :** `pandas`, `numpy`, `scikit-learn` (impliqué par `sklearn` dans les imports), `matplotlib`, `seaborn`, `plotly`.
    *   **Interface :** `streamlit` (identifié via `app.py`, bien que non explicite dans le `pyproject.toml` lu, il est requis par le code).

## 3. Architecture Détaillée du Pipeline

### 3.1. Préparation des Données (`Preparation/`)
Cette étape convertit les documents bruts (PDF, DOCX, Audio) en un format textuel uniforme (Markdown).
*   **Script :** `transformer_en_md.py`
*   **Outils :**
    *   **Docling :** Pour la conversion des documents texte (PDF, DOCX, etc.) vers Markdown.
    *   **Whisper (OpenAI) :** Pour la transcription des fichiers audio (ex: `voeux2025_trifouillis.wav`) en texte.
*   **Entrée :** Dossier `Corpus/`
*   **Sortie :** Dossier `markdown_outputs/`

### 3.2. Découpage (Chunking) (`Chunking/`)
Le texte est divisé en segments gérables pour l'indexation.
*   **Script :** `chunker.py`
*   **Stratégie Hybride :**
    1.  **Sémantique :** Découpage basé sur la structure des titres Markdown (`#`, `##`).
    2.  **Forcé (Fallback) :** Si une section est trop longue (> 1500 caractères), elle est sous-découpée avec un chevauchement (`overlap` de 200 caractères).
*   **Métadonnées :** Chaque chunk conserve son document source, et potentiellement son titre hiérarchique.
*   **Sortie :** Fichiers JSON dans `chunked_documents_by_tags/`.

### 3.3. Vectorisation (Embedding) (`Embedding/`)
Transformation des chunks textuels en vecteurs numériques.
*   **Script :** `embedder.py` (Analyse)
*   **Modèles supportés :**
    1.  **Google Gemini :** `models/text-embedding-004` (Haute performance, nécessite API Key).
    2.  **Sentence-BERT (SBERT) :** `sentence-transformers/all-MiniLM-L6-v2` (Local, rapide).
    3.  **FastText :** `cc.fr.300.bin` (Statistique, très rapide).
*   **Fonctionnalités :** Comprend des outils de visualisation des vecteurs via **t-SNE** et **Plotly** pour évaluer la qualité des clusters sémantiques.

### 3.4. Indexation (`Indexing/`)
Création de la base de données vectorielle pour la recherche rapide.
*   **Script :** `indexer.py`
*   **Technologie :** **FAISS** (Facebook AI Similarity Search).
*   **Processus :**
    *   Permet à l'utilisateur de choisir le modèle d'embedding (SBERT, Gemini, FastText).
    *   Génère les vecteurs pour tous les chunks.
    *   Sauvegarde l'index (`index.bin`) et le mapping des métadonnées (`metadata.json`) dans `faiss_indexes/`.

### 3.5. Interrogation et Génération (`Querying/`)
Le cœur du système conversationnel.
*   **Script Core :** `query_rag.py`
    *   **Classe `RAGAgent` :** Orchestre le processus.
    *   **Classification d'intention :** Distingue les questions factuelles ("SEARCH") des interactions sociales ("CHAT") pour économiser des appels RAG inutiles.
    *   **Reformulation :** Réécrit la question utilisateur en fonction de l'historique de conversation pour améliorer la recherche (Contextualisation).
    *   **Modèle de Génération :** `models/gemini-2.5-flash` (Google).
*   **Interface Utilisateur :** `app.py`
    *   Application **Streamlit**.
    *   Fonctionnalités : Historique de chat, réglages avancés (Température, Top-K, Top-P), affichage des sources avec scores de similarité, mode Debug.

### 3.6. Évaluation (`Assessment/`)
*   **Script :** `evaluate_rag.py`
*   **Données :** `test_questions.json`, `assessment_results.csv`.
*   **Méthodologie :** Utilise probablement la librairie **Ragas** (mentionnée dans `pyproject.toml`) pour calculer des métriques comme la fidélité, la pertinence de la réponse et la précision du contexte.

## 4. Points Forts Techniques

1.  **Approche Modulaire :** Séparation claire des responsabilités (Chunking, Indexing, Querying).
2.  **Flexibilité des Modèles :** Supporte à la fois des modèles locaux (SBERT, FastText) et Cloud (Gemini), permettant un arbitrage coût/performance.
3.  **Traitement Multimodal en entrée :** Capacité à ingérer de l'audio (Whisper) en plus des documents texte.
4.  **Expérience Utilisateur (UX) :**
    *   Reformulation contextuelle des questions.
    *   Classification d'intention pour des réponses plus naturelles.
    *   Interface Streamlit complète avec mode debug.

## 5. Instructions de Lancement Rapide

1.  **Installation :**
    ```bash
    poetry install
    ```
2.  **Préparation (Si nécessaire) :**
    ```bash
    poetry run python Preparation/transformer_en_md.py
    poetry run python Chunking/chunker.py
    ```
3.  **Indexation (Si nécessaire) :**
    ```bash
    poetry run python Indexing/indexer.py --method gemini
    ```
4.  **Lancement de l'Application :**
    ```bash
    poetry run streamlit run Querying/app.py
    ```
