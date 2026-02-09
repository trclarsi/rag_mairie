# 🏛️ Assistant RAG - Mairie de Triffouillis-sur-Loire

Ce projet implémente un système de **Génération Augmentée par Récupération (RAG)** pour assister les citoyens et les agents de la mairie de Triffouillis-sur-Loire. Il permet de poser des questions complexes sur les règlements municipaux, les budgets, les projets urbains et les comptes-rendus de conseils.

## 🚀 Fonctionnalités Clés

- **Multimodalité** : Ingestion de PDF, fichiers Word, CSV et même des fichiers audio (Vœux du Maire) via **Whisper**.
- **Recherche Sémantique** : Utilisation des embeddings **Gemini (text-embedding-004)** et de la bibliothèque **FAISS** pour une recherche instantanée et précise.
- **Chunking Structurel** : Découpage intelligent basé sur le format Markdown pour préserver l'unité des articles de loi.
- **Évaluation Scientifique** : Audit automatique des réponses via le framework **Ragas** avec calcul de la fidélité et de la pertinence.
- **Interface Interactive** : Application web développée avec **Streamlit**.

## 🛠️ Stack Technique

- **Langage** : Python 3.10+
- **LLM** : Kimi K2 (via Groq) & Gemini Pro
- **Embeddings** : Google Gemini API
- **Vector Store** : FAISS
- **Extraction & OCR** : Docling, EasyOCR
- **Transcription** : OpenAI Whisper
- **Interface** : Streamlit
- **Évaluation** : Ragas

## 📂 Structure du Projet

```text
├── Assessment/             # Scripts et données d'évaluation (Ragas)
│   ├── evaluate_rag.py     # Script principal d'évaluation
│   ├── test_questions.json # Set de test (Questions/Ground Truth)
│   └── assessment_results.csv # Résultats détaillés des indicateurs
├── Corpus/                 # Documents sources (PDF, DOCX, CSV, WAV)
├── Querying/               # Cœur de l'application
│   ├── app.py              # Interface Streamlit
│   ├── query_rag.py        # Agent RAG (Logique métier)
│   └── pipeline.py         # Pipeline d'ingestion et d'indexation
├── faiss_indexes/          # Index vectoriels générés
├── markdown_outputs/       # Documents convertis en Markdown pour le processing
└── publications_linkedin.md # Série de posts pour la communication projet
```

## ⚙️ Installation

1. **Cloner le projet**
2. **Installer les dépendances** :
   ```bash
   pip install -r requirements.txt  # Ou via poetry install
   ```
3. **Configurer les variables d'environnement** :
   Créer un fichier `.env` à la racine :
   ```env
   GOOGLE_API_KEY=votre_cle_gemini
   GROQ_API_KEY=votre_cle_groq
   ```

## 📖 Utilisation

### 1. Préparation des données (Ingestion & Indexation)
Lancez le pipeline pour transformer les documents du `Corpus` en index vectoriel :
```bash
python Querying/pipeline.py
```

### 2. Lancer l'Assistant (Interface Web)
Démarrez l'application Streamlit pour interagir avec le RAG :
```bash
streamlit run Querying/app.py
```

### 3. Évaluation des performances
Calculez les métriques de fidélité et de pertinence :
```bash
python Assessment/evaluate_rag.py
```

## 📈 Résultats d'Évaluation
Les derniers tests montrent un score de **Fidélité de 0.82** et une **Pertinence de 0.91**. Le système privilégie la sécurité en indiquant qu'il ne dispose pas de l'information plutôt que d'halluciner.

---
*Projet réalisé dans le cadre de la formation Deep Learning - Spécialisation RAG.*
