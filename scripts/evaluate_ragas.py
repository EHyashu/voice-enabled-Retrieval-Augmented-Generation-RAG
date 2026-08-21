import os
import json
import asyncio
from datasets import Dataset
from dotenv import load_dotenv

import sys
from unittest.mock import MagicMock
# Mock vertexai to avoid Ragas import crash with newer langchain versions
sys.modules['langchain_community.chat_models.vertexai'] = MagicMock()
sys.modules['langchain_community.llms.vertexai'] = MagicMock()

load_dotenv()

from ragas import evaluate
from ragas.metrics import (
    context_precision,
    context_recall,
    faithfulness,
    answer_relevancy
)
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings

# Import our backend modules
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from backend.vector_db import VectorDatabaseManager
import groq

def generate_answer(query: str, contexts: list[str]) -> str:
    """Generates an answer using Groq Llama 3 for evaluation purposes."""
    groq_api_key = os.getenv("GROQ_API_KEY")
    if not groq_api_key:
        return "No API Key configured."
        
    client = groq.Groq(api_key=groq_api_key)
    context_str = "\n\n".join(contexts)
    prompt = f"Context:\n{context_str}\n\nQuestion: {query}\n\nAnswer:"
    
    response = client.chat.completions.create(
        model="qwen/qwen3.6-27b",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.1,
        max_tokens=1024
    )
    return response.choices[0].message.content

async def run_evaluation():
    print("Starting Ragas Evaluation with Groq Judge...")
    
    if not os.getenv("GROQ_API_KEY"):
        print("Error: GROQ_API_KEY is not set.")
        exit(1)
        
    # Initialize Ragas Judge Wrappers
    judge_llm = ChatGroq(model="qwen/qwen3.6-27b", max_tokens=2048)
    judge_embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    
    ragas_llm = LangchainLLMWrapper(judge_llm)
    ragas_embeddings = LangchainEmbeddingsWrapper(judge_embeddings)
    
    # Initialize our FAISS Vector Database
    print("Initializing FAISS Vector Database...")
    vdb = VectorDatabaseManager()
    
    # Define Test Dataset
    test_cases = [
        {
            "question": "व्हाट इस द कैपिटल ऑफ़ राजस्थान",
            "ground_truth": "जयपुर (Jaipur)"
        },
        {
            "question": "What is the capital of France?",
            "ground_truth": "Paris"
        },
        {
            "question": "हु रोट द रामायण",
            "ground_truth": "Valmiki (वाल्मीकि)"
        }
    ]
    
    questions = []
    contexts_list = []
    answers = []
    ground_truths = []
    
    print("\nRunning RAG Retrieval & Generation for Test Cases...")
    for idx, case in enumerate(test_cases):
        q = case["question"]
        print(f"\n[{idx+1}/{len(test_cases)}] Query: {q}")
        
        # Retrieval
        search_res = vdb.search(q, top_k=2)
        matches = search_res["results"].get("matches", [])
        retrieved_contexts = [m["metadata"]["text"] for m in matches]
        
        if not retrieved_contexts:
            retrieved_contexts = ["No relevant information found."]
            
        print(f"  -> Retrieved {len(matches)} chunks.")
        
        # Generation
        ans = generate_answer(q, retrieved_contexts)
        print(f"  -> Generated Answer: {ans[:50]}...")
        
        questions.append(q)
        contexts_list.append(retrieved_contexts)
        answers.append(ans)
        ground_truths.append(case["ground_truth"])
        
    eval_data = {
        "question": questions,
        "contexts": contexts_list,
        "answer": answers,
        "ground_truth": ground_truths
    }
    
    dataset = Dataset.from_dict(eval_data)
    
    metrics = [
        context_recall,
        context_precision,
        faithfulness,
        answer_relevancy
    ]
    
    print("\nEvaluating against metrics using Groq LLM...")
    
    try:
        results = evaluate(
            dataset,
            metrics=metrics,
            llm=ragas_llm,
            embeddings=ragas_embeddings
        )
        print("\n--- Evaluation Results ---")
        for key, value in results.items():
            print(f"{key}: {value:.4f}")
            
        print("\nEvaluation completed successfully.")
    except Exception as e:
        print(f"Evaluation failed: {e}")

if __name__ == "__main__":
    asyncio.run(run_evaluation())
