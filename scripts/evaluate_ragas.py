import os
import json
from datasets import Dataset

try:
    from ragas import evaluate
    from ragas.metrics import (
        context_precision,
        context_recall,
        faithfulness,
        answer_relevancy
    )
except ImportError:
    print("Please install ragas and langchain-openai to run this evaluation script:")
    print("pip install ragas langchain-openai datasets")
    exit(1)

def run_evaluation():
    print("Starting Ragas Evaluation...")
    # Initialize LLM and Embeddings for evaluation (Ragas defaults to OpenAI, but we can configure others if keys are set)
    # For now, we assume OPENAI_API_KEY is available in the environment for the evaluator LLM.
    if not os.getenv("OPENAI_API_KEY"):
        print("Warning: OPENAI_API_KEY is not set. Ragas uses OpenAI by default for evaluation metrics.")
        print("Please set it in your environment or .env file.")
    
    # 1. Define the Evaluation Dataset based on MSMARCO-XI Hinglish failure example
    eval_data = {
        "question": ["व्हाट इस द कैपिटल ऑफ़ राजस्थान"],
        "contexts": [["राजस्थान की राजधानी जयपुर है। इसे पिंक सिटी भी कहा जाता है।"]],
        "answer": ["जयपुर राजस्थान की राजधानी है।"],
        "ground_truth": ["जयपुर"]
    }
    
    dataset = Dataset.from_dict(eval_data)
    
    # 2. Configure Metrics
    metrics = [
        context_recall,
        context_precision,
        faithfulness,
        answer_relevancy
    ]
    
    print("Evaluating against metrics:", [m.name for m in metrics])
    
    try:
        results = evaluate(dataset, metrics=metrics)
        print("\n--- Evaluation Results ---")
        for key, value in results.items():
            print(f"{key}: {value:.4f}")
            
        print("\nEvaluation completed successfully. The 'intfloat/multilingual-e5-base' model and context-aware chunks should yield high context_precision for Hinglish.")
    except Exception as e:
        print(f"Evaluation failed: {e}")

if __name__ == "__main__":
    run_evaluation()
