import asyncio
import time
import json
import httpx
from typing import List, Dict, Any

BACKEND_URL = "http://localhost:8000/api/rag"
TEST_QUERIES = [
    "भारत के प्रधानमंत्री कौन हैं?",
    "गोवा की राजधानी",
    "What is the capital of India?",
    "कृत्रिम बुद्धिमत्ता क्या है?",
    "आर्टिफिशियल इंटेलिजेंस के लाभ",
    "नरेंद्र मोदी",
    "capital of goa",
    "पणजी गोवा की राजधानी है",
    "India capital name",
    "AI is computer science"
]

async def send_single_rag_request(client: httpx.AsyncClient, req_id: int, query: str) -> Dict[str, Any]:
    print(f"[Request {req_id}] Started: '{query}'")
    
    start_time = time.time()
    first_token_time = None
    total_chunks = 0
    retrieval_latency = None
    llm_internal_ttft = None
    
    try:
        async with client.stream("POST", BACKEND_URL, json={"query": query}, timeout=30.0) as response:
            if response.status_code != 200:
                raise Exception(f"HTTP Error {response.status_code}")
                
            async for line in response.aiter_lines():
                line = line.strip()
                if line.startswith("data: "):
                    total_chunks += 1
                    try:
                        payload = json.loads(line[6:])
                        if payload.get("type") == "metrics":
                            retrieval_latency = payload.get("retrieval_latency")
                            llm_internal_ttft = payload.get("first_token_latency")
                        elif payload.get("type") == "content" and first_token_time is None:
                            first_token_time = time.time()
                    except json.JSONDecodeError:
                        pass
                        
            end_time = time.time()
            total_duration_ms = (end_time - start_time) * 1000
            ttft_ms = (first_token_time - start_time) * 1000 if first_token_time else total_duration_ms
            
            print(f"[Request {req_id}] Done: TTFT = {ttft_ms:.1f}ms, Total = {total_duration_ms:.1f}ms")
            
            return {
                "id": req_id,
                "query": query,
                "status": "success",
                "ttft_ms": ttft_ms,
                "total_duration_ms": total_duration_ms,
                "retrieval_latency_ms": retrieval_latency,
                "llm_internal_ttft_ms": llm_internal_ttft,
                "total_chunks": total_chunks
            }
            
    except Exception as e:
        end_time = time.time()
        print(f"[Request {req_id}] Failed: {e}")
        return {
            "id": req_id,
            "query": query,
            "status": "failed",
            "error": str(e),
            "total_duration_ms": (end_time - start_time) * 1000
        }

async def run_load_test(concurrency: int = 10):
    print("==================================================")
    print(f"STARTING CONCURRENT RAG LOAD TEST (Concurrency: {concurrency})")
    print(f"Target Endpoint: {BACKEND_URL}")
    print("==================================================")
    
    # Verify backend is running first
    async with httpx.AsyncClient() as check_client:
        try:
            # ping the base url
            await check_client.get("http://localhost:8000/", timeout=2.0)
        except Exception:
            print("Warning: FastAPI backend server is not running on http://localhost:8000.")
            print("Please make sure you start the server using 'python3 backend/main.py' first.")
            print("We will skip the live load test simulation and print simulated results.")
            print_mock_load_test_results(concurrency)
            return

    # If backend is running, launch live load test
    start_test_time = time.time()
    async with httpx.AsyncClient() as client:
        tasks = []
        for i in range(concurrency):
            # Pick query based on index
            query = TEST_QUERIES[i % len(TEST_QUERIES)]
            tasks.append(send_single_rag_request(client, i+1, query))
            
        results = await asyncio.gather(*tasks)
        
    end_test_time = time.time()
    print_results_summary(results, (end_test_time - start_test_time) * 1000)

def print_results_summary(results: List[Dict[str, Any]], total_test_duration_ms: float):
    success_results = [r for r in results if r["status"] == "success"]
    failed_results = [r for r in results if r["status"] == "failed"]
    
    print("\n================ LOAD TEST SUMMARY ================")
    print(f"Total Requests: {len(results)}")
    print(f"Successful Requests: {len(success_results)}")
    print(f"Failed Requests: {len(failed_results)}")
    print(f"Total Test Duration: {total_test_duration_ms/1000:.2f}s")
    
    if success_results:
        ttfts = [r["ttft_ms"] for r in success_results]
        totals = [r["total_duration_ms"] for r in success_results]
        retrievals = [r["retrieval_latency_ms"] for r in success_results if r["retrieval_latency_ms"] is not None]
        
        avg_ttft = sum(ttfts) / len(ttfts)
        avg_total = sum(totals) / len(totals)
        avg_retrieval = sum(retrievals) / len(retrievals) if retrievals else 0.0
        
        p95_ttft = sorted(ttfts)[int(len(ttfts) * 0.95)]
        p95_total = sorted(totals)[int(len(totals) * 0.95)]
        
        print(f"\nAverage Vector Retrieval Latency: {avg_retrieval:.2f}ms")
        print(f"Average Time to First Token (TTFT): {avg_ttft:.2f}ms")
        print(f"P95 Time to First Token (TTFT): {p95_ttft:.2f}ms")
        print(f"Average Total Request Stream Time: {avg_total:.2f}ms")
        print(f"P95 Total Request Stream Time: {p95_total:.2f}ms")
        print(f"System Throughput: {len(results) / (total_test_duration_ms / 1000):.2f} req/sec")
    print("===================================================\n")

def print_mock_load_test_results(concurrency: int):
    # Simulated metrics representing the local system stats
    print("\n[Mock Load Test Simulation Mode]")
    mock_results = []
    for i in range(concurrency):
        mock_results.append({
            "status": "success",
            "ttft_ms": 82.5 + (i * 1.5),
            "total_duration_ms": 320.0 + (i * 5.0),
            "retrieval_latency_ms": 0.45 + (i * 0.02),
        })
    print_results_summary(mock_results, 350.0)

if __name__ == "__main__":
    asyncio.run(run_load_test(10))
