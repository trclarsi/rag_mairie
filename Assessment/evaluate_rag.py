
import os
import json
import pandas as pd
import time
from datasets import Dataset
from ragas import evaluate
from ragas.run_config import RunConfig
from dotenv import load_dotenv

# Charger les variables d'environnement depuis .env
load_dotenv()

from ragas.metrics import (
    faithfulness,
    answer_relevancy,
    context_precision,
    context_recall,
)
from langchain_groq import ChatGroq
from langchain_google_genai import GoogleGenerativeAIEmbeddings

# Import de votre agent
import sys
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PARENT_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, '..'))
sys.path.append(os.path.join(PARENT_DIR, 'Querying'))
from query_rag import RAGAgent

# ==============================================================================
# CONFIGURATION
# ==============================================================================

TEST_FILE = os.path.join(SCRIPT_DIR, "test_questions.json")
KIMI_MODEL = "moonshotai/kimi-k2-instruct-0905"

def run_assessment():
    print("\n" + "="*60)
    print("🚀 ÉVALUATION RAG : JUGE = KIMI K2 | EMBEDDINGS = GEMINI")
    print("="*60)
    
    google_key = os.environ.get("GOOGLE_API_KEY")
    groq_key = os.environ.get("GROQ_API_KEY")
    
    if not google_key or not groq_key:
        print("❌ Erreur : Clés API manquantes.")
        return

    # Juge LLM
    evaluator_llm = ChatGroq(model=KIMI_MODEL, groq_api_key=groq_key, temperature=0)
    
    # Embeddings (Nom du modèle corrigé pour LangChain)
    evaluator_embeddings = GoogleGenerativeAIEmbeddings(
        model="models/gemini-embedding-001", 
        google_api_key=google_key
    )
    
    # 1. Charger les tests
    with open(TEST_FILE, 'r', encoding='utf-8') as f:
        test_data = json.load(f)
    
    # 2. Init Agent
    agent = RAGAgent()

    # 3. Collecte
    questions, answers, contexts, ground_truths = [], [], [], []
    print(f"\n📝 Phase 1 : Collecte des réponses...")

    for i, item in enumerate(test_data):
        print(f"   [{i+1}/{len(test_data)}] Question : {item['question']}")
        answer, docs = agent.ask(item['question'])
        questions.append(item['question'])
        answers.append(answer)
        contexts.append([doc.get('content', '') for doc in docs])
        ground_truths.append(item['ground_truth'])
        time.sleep(1)

    # 4. Dataset
    dataset = Dataset.from_dict({
        "question": questions, "answer": answers, 
        "contexts": contexts, "ground_truth": ground_truths
    })

    # 5. Évaluation avec RunConfig pour éviter les 429
    print("\n⚖️ Phase 2 : Calcul des métriques Ragas...")
    
    try:
        # max_workers=1 et thread_timeout pour stabiliser Groq
        config = RunConfig(max_workers=1, timeout=60)
        
        result = evaluate(
            dataset,
            metrics=[faithfulness, answer_relevancy, context_precision, context_recall],
            llm=evaluator_llm,
            embeddings=evaluator_embeddings,
            run_config=config
        )

        # 6. Résultats
        print("\n" + "="*60)
        print("📊 RÉSULTATS DE L'ÉVALUATION")
        print("="*60)
        
        df = result.to_pandas()
        
        # Identification des colonnes de métriques
        metric_cols = ['faithfulness', 'answer_relevancy', 'context_precision', 'context_recall']
        # S'assurer que les colonnes existent (Ragas peut changer les noms selon la version)
        actual_metrics = [c for c in metric_cols if c in df.columns]
        
        # Calcul des moyennes pour la ligne de résumé
        summary = df[actual_metrics].mean()
        summary_df = pd.DataFrame([["MOYENNE GÉNÉRALE"] + ["-"] * (len(df.columns) - len(actual_metrics) - 1) + summary.tolist()], 
                                  columns=df.columns)
        
        # Concaténer les résultats avec la ligne de résumé
        final_df = pd.concat([df, summary_df], ignore_index=True)
        
        # Sauvegarde
        timestamp = time.strftime("%Y%m%d-%H%M%S")
        output_csv = os.path.join(SCRIPT_DIR, "assessment_results.csv")
        history_csv = os.path.join(SCRIPT_DIR, f"assessment_results_{timestamp}.csv")
        
        final_df.to_csv(output_csv, index=False, encoding='utf-8-sig')
        final_df.to_csv(history_csv, index=False, encoding='utf-8-sig')
        
        print(f"\n✅ Rapport sauvegardé : {output_csv}")
        print(f"✅ Historique sauvegardé : {history_csv}")
        print(f"\nScores moyens :\n{summary}")

    except Exception as e:
        print(f"❌ Erreur lors de l'évaluation : {e}")

if __name__ == "__main__":
    run_assessment()
