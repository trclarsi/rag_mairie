import os
import json
import pandas as pd
from datasets import Dataset
from ragas import evaluate
from ragas.metrics import (
    Faithfulness,
    AnswerRelevancy,
    ContextPrecision,
    ContextRecall,
)
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings

# Import de votre agent
import sys
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PARENT_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, '..'))
sys.path.append(os.path.join(PARENT_DIR, 'Querying'))
from query_rag import RAGAgent

# ==============================================================================
# CONFIGURATION
# ==============================================================================

# Fichier de test
TEST_FILE = os.path.join(SCRIPT_DIR, "test_questions.json")

def run_assessment():
    print("🚀 Démarrage de l'évaluation RAG...")
    
    # 0. Configuration de la clé API
    api_key = os.environ.get("GOOGLE_API_KEY")
    if not api_key:
        print("⚠️ Variable GOOGLE_API_KEY non trouvée.")
        api_key = input("Veuillez entrer votre clé API Google Gemini pour l'évaluation : ").strip()
        os.environ["GOOGLE_API_KEY"] = api_key

    # Initialisation des modèles pour l'évaluation (Ragas a besoin d'un LLM "juge")
    evaluator_llm = ChatGoogleGenerativeAI(
        model="gemini-2.0-flash", 
        google_api_key=api_key
    )
    evaluator_embeddings = GoogleGenerativeAIEmbeddings(
        model="models/text-embedding-004", 
        google_api_key=api_key
    )
    
    # 1. Charger les questions de test
    with open(TEST_FILE, 'r', encoding='utf-8') as f:
        test_data = json.load(f)
    
    # 2. Initialiser l'agent RAG
    try:
        agent = RAGAgent()
    except Exception as e:
        print(f"❌ Erreur d'initialisation de l'agent : {e}")
        return

    # 3. Collecter les réponses du RAG
    questions = []
    answers = []
    contexts = []
    ground_truths = []

    for item in test_data:
        question = item['question']
        print(f"🔍 Test de la question : {question}")
        
        # Obtenir la réponse et les documents sources
        answer, docs = agent.ask(question)
        
        questions.append(question)
        answers.append(answer)
        # Ragas attend une liste de listes de strings pour les contextes
        contexts.append([doc['content'] for doc in docs])
        ground_truths.append(item['ground_truth'])

    # 4. Préparer le dataset pour Ragas
    data_dict = {
        "question": questions,
        "answer": answers,
        "contexts": contexts,
        "ground_truth": ground_truths
    }
    dataset = Dataset.from_dict(data_dict)

    # 5. Lancer l'évaluation
    print("\n⚖️ Calcul des métriques Ragas (Gemini est le juge)...")
    
    result = evaluate(
        dataset,
        metrics=[
            Faithfulness(),
            AnswerRelevancy(),
            ContextPrecision(),
            ContextRecall(),
        ],
        llm=evaluator_llm,
        embeddings=evaluator_embeddings
    )

    # 6. Afficher et sauvegarder les résultats
    print("\n" + "="*60)
    print("📊 RÉSULTATS DE L'ÉVALUATION")
    print("="*60)
    df = result.to_pandas()
    # Affichage des colonnes disponibles
    cols_to_show = ['question', 'faithfulness', 'answer_relevancy', 'context_precision', 'context_recall']
    # On ne garde que les colonnes qui existent réellement dans le DF pour éviter les erreurs
    existing_cols = [c for c in cols_to_show if c in df.columns]
    print(df[existing_cols])
    
    output_csv = os.path.join(SCRIPT_DIR, "assessment_results.csv")
    df.to_csv(output_csv, index=False)
    print(f"\n✅ Résultats complets sauvegardés dans : {output_csv}")
    print(f"Scores moyens : \n{result}")

if __name__ == "__main__":
    run_assessment()