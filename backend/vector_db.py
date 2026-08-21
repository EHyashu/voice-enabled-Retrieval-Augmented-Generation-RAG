import json
import os
import time
from typing import Any

import numpy as np
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_INDEX_NAME = os.getenv("PINECONE_INDEX_NAME", "msmarco-rag-e5")

# Class for fallback Local Vector Database (in case Pinecone key is missing or invalid)
import pickle


class LocalVectorDB:
    def __init__(self, storage_path="data/local_vectors.pkl"):
        print("Using local mock vector database (fallback with FAISS)...")
        self.storage_path = storage_path
        self.vectors = {}
        self.id_map = []
        import faiss
        self.index = faiss.IndexFlatIP(384) # MiniLM outputs 384 dims
        self.load_from_disk()
        
    def build_faiss_index(self):
        import faiss
        self.index = faiss.IndexFlatIP(384)
        self.id_map = []
        
        if not self.vectors:
            return
            
        vectors_list = []
        for vid, data in self.vectors.items():
            vectors_list.append(data["values"])
            self.id_map.append(vid)
            
        if vectors_list:
            vec_array = np.array(vectors_list, dtype=np.float32)
            faiss.normalize_L2(vec_array)
            self.index.add(vec_array)

    def load_from_disk(self):
        if os.path.exists(self.storage_path):
            try:
                with open(self.storage_path, "rb") as f:
                    self.vectors = pickle.load(f)
                print(f"Loaded {len(self.vectors)} vectors from local storage.")
                self.build_faiss_index()
            except Exception as e:
                print(f"Failed to load local vectors: {e}")

    def save_to_disk(self):
        os.makedirs(os.path.dirname(self.storage_path), exist_ok=True)
        with open(self.storage_path, "wb") as f:
            pickle.dump(self.vectors, f)
            
    def upsert(self, vectors: list[dict[str, Any]]):
        for v in vectors:
            self.vectors[v["id"]] = {
                "values": np.array(v["values"], dtype=np.float32),
                "metadata": v["metadata"]
            }
        self.save_to_disk()
        self.build_faiss_index()
        print(f"Upserted {len(vectors)} vectors locally.")
        
    def query(self, vector: list[float], top_k: int = 5, include_metadata: bool = True) -> dict[str, Any]:
        if not self.vectors or self.index.ntotal == 0:
            return {"matches": []}
            
        import faiss
        query_vec = np.array([vector], dtype=np.float32)
        faiss.normalize_L2(query_vec)
        
        # Search FAISS index
        scores, indices = self.index.search(query_vec, min(top_k, self.index.ntotal))
        
        results = []
        for i, idx in enumerate(indices[0]):
            if idx == -1: continue
            vid = self.id_map[idx]
            data = self.vectors[vid]
            results.append({
                "id": vid,
                "score": float(scores[0][i]),
                "metadata": data["metadata"] if include_metadata else {}
            })
            
        return {"matches": results}

class VectorDatabaseManager:
    def __init__(self):
        self.embedding_model = None
        self.pc = None
        self.index = None
        self.use_fallback = False
        self.kv_store = {}
        
        # Load local KV store
        self.kv_store_path = "data/kv_store.json"
        if os.path.exists(self.kv_store_path):
            try:
                with open(self.kv_store_path, "r", encoding="utf-8") as f:
                    self.kv_store = json.load(f)
            except Exception as e:
                print(f"Failed to load KV store: {e}")
        
        # Load embedding model
        self.init_embedding_model()
        self.init_redis()
        
        # Force Local Vector DB to eliminate 4000ms+ network spikes from Pinecone serverless
        print("Forcing LocalVectorDB for zero network latency.")
        self.index = LocalVectorDB()
        self.use_fallback = True
        
    def init_redis(self):
        import redis
        try:
            self.redis_client = redis.Redis(host='localhost', port=6379, db=0, decode_responses=True, socket_connect_timeout=0.5)
            self.redis_client.ping()
            print("Connected to Redis cache.")
        except Exception:
            print("Redis not available, using in-memory dictionary cache.")
            self.redis_client = None
        self.local_cache = {}

    def init_embedding_model(self):
        print("Loading sentence-transformers/all-MiniLM-L6-v2 embedding model...")
        from sentence_transformers import SentenceTransformer
        # Force CPU because MPS on Mac has high initialization latency for single embeddings
        device = "cpu"
        print(f"Using device: {device} for local inference.")
        # all-MiniLM-L6-v2 is ultra fast for English RAG, outputs 384-dimensional vectors
        self.embedding_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2", device=device)
        print("Warming up embedding model to prevent cold start latency...")
        self.embedding_model.encode(["warmup"], show_progress_bar=False)
        print("Embedding model loaded successfully.")

    def get_embeddings(self, texts: list[str]) -> list[list[float]]:
        embeddings = self.embedding_model.encode(texts, show_progress_bar=False, normalize_embeddings=True)
        return embeddings.tolist()

    def upsert_chunks(self, chunks: list[dict[str, Any]], batch_size: int = 32):
        print(f"Upserting {len(chunks)} chunks in batches of {batch_size}...")
        
        for i in range(0, len(chunks), batch_size):
            batch = chunks[i:i + batch_size]
            # MiniLM models don't require prefix
            texts = [c["text"] for c in batch]
            embeddings = self.get_embeddings(texts)
            
            vectors_to_upsert = []
            for chunk, emb in zip(batch, embeddings):
                # Save text to local KV store
                self.kv_store[chunk["id"]] = chunk["text"]
                
                # Exclude large text payload from Pinecone metadata to ensure fast retrieval
                metadata = {
                    "strategy": chunk["strategy"],
                    "source_id": chunk["metadata"].get("source_id", ""),
                    "passage_index": chunk["metadata"].get("passage_index", 0),
                    "query": chunk["metadata"].get("query", ""),
                    "is_selected": int(chunk["metadata"].get("is_selected", 0))
                }
                vectors_to_upsert.append({
                    "id": chunk["id"],
                    "values": emb,
                    "metadata": metadata
                })
            
            self.index.upsert(vectors=vectors_to_upsert)
            
        # Flush KV store to disk
        os.makedirs("data", exist_ok=True)
        with open(self.kv_store_path, "w", encoding="utf-8") as f:
            json.dump(self.kv_store, f, ensure_ascii=False, indent=2)
            
        print("Upsert completed successfully.")

    def search(self, query: str, top_k: int = 5) -> dict[str, Any]:
        import hashlib
        import json
        cache_key = f"emb:{hashlib.sha256(query.strip().lower().encode()).hexdigest()}"
        
        query_embedding = None
        if self.redis_client:
            cached = self.redis_client.get(cache_key)
            if cached: query_embedding = json.loads(cached)
        else:
            query_embedding = self.local_cache.get(cache_key)
            
        if not query_embedding:
            # MiniLM models don't require prefix
            query_embedding = self.embedding_model.encode([query], normalize_embeddings=True)[0].tolist()
            if self.redis_client:
                self.redis_client.set(cache_key, json.dumps(query_embedding), ex=3600)
            else:
                self.local_cache[cache_key] = query_embedding
        
        start_time = time.time()
        if self.use_fallback:
            results = self.index.query(vector=query_embedding, top_k=top_k, include_metadata=True)
        else:
            results = self.index.query(vector=query_embedding, top_k=top_k, include_metadata=True)
            
        # Re-inject text from local KV store
        for match in results.get("matches", []):
            if "metadata" not in match:
                match["metadata"] = {}
            match["metadata"]["text"] = self.kv_store.get(match["id"], "[Text not found in KV store]")
            
        latency_ms = (time.time() - start_time) * 1000
        
        return {
            "results": results,
            "latency_ms": latency_ms
        }

def run_benchmarks():
    print("--- RUNNING VECTOR DB RETRIEVAL BENCHMARKS ---")
    vdb = VectorDatabaseManager()
    
    # Check if chunks.json exists
    chunks_path = "data/chunks.json"
    if not os.path.exists(chunks_path):
        print(f"Error: {chunks_path} not found. Please run chunking_strategies.py first.")
        return
        
    with open(chunks_path, "r", encoding="utf-8") as f:
        chunks = json.load(f)
        
    # Upsert chunks
    vdb.upsert_chunks(chunks)
    
    # Test queries
    test_queries = [
        "भारत के प्रधानमंत्री कौन हैं?",
        "गोवा की राजधानी",
        "What is the capital of India?",
        "कृत्रिम बुद्धिमत्ता के फायदे"
    ]
    
    latencies = []
    print("\nRetrieving results for test queries:")
    for q in test_queries:
        res = vdb.search(q, top_k=3)
        latency = res["latency_ms"]
        latencies.append(latency)
        print(f"\nQuery: '{q}'")
        print(f"Latency: {latency:.2f}ms")
        for match in res["results"]["matches"]:
            score = match.get("score", 0.0)
            text = match.get("metadata", {}).get("text", "")
            strategy = match.get("metadata", {}).get("strategy", "")
            print(f" - [{strategy}] (Score: {score:.4f}): {text[:100]}...")
            
    avg_latency = sum(latencies) / len(latencies)
    print(f"\nAverage Retrieval Latency: {avg_latency:.2f}ms")
    if avg_latency < 50:
        print("Success: Retrieval latency is under the 50ms budget! ✓")
    else:
        print("Warning: Retrieval latency exceeds the 50ms budget.")

if __name__ == "__main__":
    run_benchmarks()
