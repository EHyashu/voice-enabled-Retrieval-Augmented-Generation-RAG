import os
import json
import time
from typing import Dict, Any, List
from fastapi import FastAPI, Request, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from dotenv import load_dotenv

# Import vector database manager
from backend.vector_db import VectorDatabaseManager
from backend.ingestion import DocumentIngestor

load_dotenv()

app = FastAPI(title="Voice-Enabled RAG Backend")

# Enable CORS for frontend requests
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify front-end URL
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize Vector DB manager
vdb_manager = VectorDatabaseManager()

# Groq Client Setup
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
groq_client = None
if GROQ_API_KEY and GROQ_API_KEY.strip() != "" and GROQ_API_KEY != "your_groq_api_key_here":
    try:
        from groq import Groq
        groq_client = Groq(api_key=GROQ_API_KEY)
        print("Groq client initialized successfully.")
    except Exception as e:
        print(f"Failed to initialize Groq client ({e}).")

# Anthropic Client Setup
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")
anthropic_client = None
if ANTHROPIC_API_KEY and ANTHROPIC_API_KEY.strip() != "" and ANTHROPIC_API_KEY != "your_anthropic_api_key_here":
    try:
        from anthropic import Anthropic
        anthropic_client = Anthropic(api_key=ANTHROPIC_API_KEY)
        print("Anthropic client initialized successfully.")
    except Exception as e:
        print(f"Failed to initialize Anthropic client ({e}).")

# Determine if we should use Mock fallback
use_mock_llm = (groq_client is None and anthropic_client is None)
if use_mock_llm:
    print("Neither Groq nor Anthropic API key is configured. Using Mock LLM responder.")

class RAGRequest(BaseModel):
    query: str

def get_mock_stream(query: str, sources: List[Dict[str, Any]], retrieval_latency: float):
    """Generates a mock streaming answer in case Claude API is not configured."""
    source_texts = "\n".join([f"- [{s['metadata']['strategy']}] {s['metadata']['text']}" for s in sources])
    
    response_template = f"नमस्कार! (Mock LLM response for: '{query}')\n\nBased on the retrieved context, here is what I found:\n\n"
    if sources:
        response_template += f"Relevant context found via vector retrieval ({retrieval_latency:.2f}ms):\n{source_texts}\n\nThis is a mock answer stream because the Anthropic API Key is not configured. Setup ANTHROPIC_API_KEY in your .env file to enable Claude 3.5 Sonnet response generation."
    else:
        response_template += "No sources were found in the database. Please make sure to run the chunking and vector index ingestion scripts first."

    # Yield sources first
    yield "data: " + json.dumps({'type': 'sources', 'sources': [{'text': s['metadata']['text'], 'strategy': s['metadata']['strategy'], 'score': s.get('score', 0)} for s in sources]}) + "\n\n"
    
    # Perceived latency delay simulation
    start_time = time.time()
    time.sleep(0.08) # Simulate LLM first token latency (80ms)
    first_token_latency = (time.time() - start_time) * 1000
    
    yield "data: " + json.dumps({'type': 'metrics', 'retrieval_latency': retrieval_latency, 'first_token_latency': first_token_latency}) + "\n\n"
    
    # Yield content chunk by chunk
    words = response_template.split(" ")
    for word in words:
        yield "data: " + json.dumps({'type': 'content', 'content': word + ' '}) + "\n\n"
        time.sleep(0.02) # simulate stream
        
    # Yield final overall metrics
    yield "data: " + json.dumps({'type': 'done'}) + "\n\n"

ingestor = DocumentIngestor()

@app.post("/api/upload")
async def upload_document(file: UploadFile = File(...)):
    print(f"Received file upload: {file.filename}")
    try:
        contents = await file.read()
        chunks = ingestor.process_document(contents, file.filename)
        
        # Upsert the chunks into our vector database
        vdb_manager.upsert_chunks(chunks)
        
        return {"status": "success", "message": f"Successfully ingested {len(chunks)} chunks from {file.filename}."}
    except Exception as e:
        print(f"Error processing document: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/rag")
async def rag_pipeline(req: RAGRequest):
    query = req.query
    print(f"Processing RAG query: '{query}'")
    
    # 1. Retrieve top-3 from Pinecone / Local Vector DB
    retrieval_start = time.time()
    search_res = vdb_manager.search(query, top_k=3)
    retrieval_latency = (time.time() - retrieval_start) * 1000
    
    sources = search_res["results"].get("matches", [])
    print(f"Retrieved {len(sources)} sources in {retrieval_latency:.2f}ms")
    
    highest_score = max([s.get('score', 0) for s in sources]) if sources else 0

    if use_mock_llm:
        return StreamingResponse(
            get_mock_stream(query, sources, retrieval_latency),
            media_type="text/event-stream"
        )
        
    # Construct LLM prompt with context
    context_str = "\n\n".join([
        f"Source [{i+1}] (Match Score: {s.get('score', 0):.1%}, Strategy: {s['metadata']['strategy']}):\n{s['metadata']['text']}"
        for i, s in enumerate(sources)
    ])
    
    system_prompt = (
        "You are a voice assistant for the Voice RAG Orchestrator.\n"
        "Rules:\n"
        "- Answer in 1-2 short sentences maximum.\n"
        "- Use only relevant context.\n"
        "- Don't mention sources.\n"
        "- Don't use markdown.\n"
        "- If context is insufficient, say so.\n"
        "- Be conversational."
    )
    
    user_prompt = f"Context:\n{context_str}\n\nQuestion: {query}\n\nConcise Answer:"
    
    def generate_llm_stream():
        # First send the sources to the client
        yield "data: " + json.dumps({'type': 'sources', 'sources': [{'text': s['metadata']['text'], 'strategy': s['metadata']['strategy'], 'score': s.get('score', 0)} for s in sources]}) + "\n\n"
        
        start_time = time.time()
        first_token_sent = False
        first_token_latency = 0.0
        
        # 1. Try Groq API (Llama 3.3) for blazing fast TTFT
        if groq_client:
            try:
                print("Attempting Groq API stream...", flush=True)
                system_prompt_with_context = f"""{system_prompt}
CRITICAL INSTRUCTION: You MUST reply in English ONLY. Do NOT use Hindi script or any other language, even if the user asks in Hindi. This is strictly required for the Text-to-Speech engine to work. Keep your answer brief and conversational.

Context Information:
{context_str}"""
                
                completion = groq_client.chat.completions.create(
                    model="openai/gpt-oss-20b",
                    messages=[
                        {"role": "system", "content": system_prompt_with_context},
                        {"role": "user", "content": user_prompt}
                    ],
                    temperature=0.3,
                    max_tokens=150,
                    stream=True,
                    timeout=10.0
                )
                buffer = ""
                in_think_block = False
                
                for chunk in completion:
                    text_chunk = chunk.choices[0].delta.content
                    if text_chunk:
                        buffer += text_chunk
                        
                        while True:
                            if not in_think_block:
                                think_start = buffer.find("<think>")
                                if think_start != -1:
                                    # Yield everything before <think>
                                    safe_text = buffer[:think_start]
                                    if safe_text:
                                        if not first_token_sent:
                                            first_token_latency = (time.time() - start_time) * 1000
                                            first_token_sent = True
                                            yield "data: " + json.dumps({'type': 'metrics', 'retrieval_latency': retrieval_latency, 'first_token_latency': first_token_latency}) + "\n\n"
                                        yield "data: " + json.dumps({'type': 'content', 'content': safe_text}) + "\n\n"
                                    
                                    # Move buffer past <think>
                                    buffer = buffer[think_start + 7:]
                                    in_think_block = True
                                    continue
                                else:
                                    # If there's a '<' at the very end of the buffer, it might be the start of '<think>'. We should hold it.
                                    # Find the last '<'
                                    last_open = buffer.rfind("<")
                                    if last_open != -1 and "<think>".startswith(buffer[last_open:]):
                                        # Yield everything before the potential tag
                                        safe_text = buffer[:last_open]
                                        if safe_text:
                                            if not first_token_sent:
                                                first_token_latency = (time.time() - start_time) * 1000
                                                first_token_sent = True
                                                yield "data: " + json.dumps({'type': 'metrics', 'retrieval_latency': retrieval_latency, 'first_token_latency': first_token_latency}) + "\n\n"
                                            yield "data: " + json.dumps({'type': 'content', 'content': safe_text}) + "\n\n"
                                        buffer = buffer[last_open:]
                                    else:
                                        # Safe to yield whole buffer
                                        if buffer:
                                            if not first_token_sent:
                                                first_token_latency = (time.time() - start_time) * 1000
                                                first_token_sent = True
                                                yield "data: " + json.dumps({'type': 'metrics', 'retrieval_latency': retrieval_latency, 'first_token_latency': first_token_latency}) + "\n\n"
                                            yield "data: " + json.dumps({'type': 'content', 'content': buffer}) + "\n\n"
                                        buffer = ""
                                    break
                            else:
                                think_end = buffer.find("</think>")
                                if think_end != -1:
                                    buffer = buffer[think_end + 8:]
                                    in_think_block = False
                                    continue
                                else:
                                    # Might be partially forming </think>
                                    last_open = buffer.rfind("<")
                                    if last_open != -1 and "</think>".startswith(buffer[last_open:]):
                                        buffer = buffer[last_open:]
                                    else:
                                        buffer = ""
                                    break

                if buffer and not in_think_block and not "<think>".startswith(buffer):
                    if not first_token_sent:
                        first_token_latency = (time.time() - start_time) * 1000
                        first_token_sent = True
                        yield "data: " + json.dumps({'type': 'metrics', 'retrieval_latency': retrieval_latency, 'first_token_latency': first_token_latency}) + "\n\n"
                    yield "data: " + json.dumps({'type': 'content', 'content': buffer}) + "\n\n"
                
                yield "data: " + json.dumps({'type': 'done'}) + "\n\n"
                return
            except Exception as groq_err:
                print(f"Groq API error encountered: {groq_err}. Falling back to Anthropic...", flush=True)
                
        # 2. Try Anthropic Claude API
        if anthropic_client:
            try:
                print("Attempting Anthropic Claude API stream...", flush=True)
                with anthropic_client.messages.stream(
                    model="claude-3-5-sonnet-20241022",
                    max_tokens=150,
                    system=system_prompt,
                    messages=[{"role": "user", "content": user_prompt}],
                    timeout=1.0
                ) as stream:
                    for text_chunk in stream.text_stream:
                        if not first_token_sent:
                            first_token_latency = (time.time() - start_time) * 1000
                            first_token_sent = True
                            yield "data: " + json.dumps({'type': 'metrics', 'retrieval_latency': retrieval_latency, 'first_token_latency': first_token_latency}) + "\n\n"
                        yield "data: " + json.dumps({'type': 'content', 'content': text_chunk}) + "\n\n"
                
                yield "data: " + json.dumps({'type': 'done'}) + "\n\n"
                return
            except Exception as anth_err:
                print(f"Anthropic API error encountered: {anth_err}. Falling back to mock...", flush=True)
                
        # 3. Final Fallback: Mock Response stream showing context facts
        print("Using local mock LLM stream as final fallback...", flush=True)
        fallback_msg = "Note: LLM providers are currently rate-limited. Summarized context:\n"
        fallback_msg += "\n".join([f"- {s['metadata']['text']}" for s in sources]) if sources else "- No relevant context found."
        
        if not first_token_sent:
            first_token_latency = 80.0
            yield "data: " + json.dumps({'type': 'metrics', 'retrieval_latency': retrieval_latency, 'first_token_latency': first_token_latency}) + "\n\n"
            
        words = fallback_msg.split(" ")
        for word in words:
            yield "data: " + json.dumps({'type': 'content', 'content': word + ' '}) + "\n\n"
            time.sleep(0.02)
            
        yield "data: " + json.dumps({'type': 'done'}) + "\n\n"
            
    return StreamingResponse(
        generate_llm_stream(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache, no-transform",
            "Connection": "keep-alive",
            "X-Content-Type-Options": "nosniff"
        }
    )

if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 8000))
    host = os.getenv("HOST", "0.0.0.0")
    uvicorn.run(app, host=host, port=port)
