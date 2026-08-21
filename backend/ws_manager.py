import asyncio
import json
import os
import time
from typing import Dict, Any, List
from fastapi import WebSocket, WebSocketDisconnect
from deepgram import (
    DeepgramClient,
    LiveTranscriptionEvents,
    LiveOptions,
    SpeakOptions,
)
from backend.vector_db import VectorDatabaseManager
import groq
import httpx
from anthropic import Anthropic

class ConnectionManager:
    def __init__(self, vdb_manager: VectorDatabaseManager):
        self.active_connections: list[WebSocket] = []
        self.vdb_manager = vdb_manager
        
        self.dg_key = os.getenv("DEEPGRAM_API_KEY")
        if self.dg_key:
            self.deepgram = DeepgramClient(self.dg_key)
        else:
            self.deepgram = None
            print("WARNING: DEEPGRAM_API_KEY not found.")
            
        groq_api_key = os.getenv("GROQ_API_KEY")
        self.groq_client = groq.Groq(api_key=groq_api_key) if groq_api_key else None
        
        anthropic_key = os.getenv("ANTHROPIC_API_KEY")
        self.anthropic_client = Anthropic(api_key=anthropic_key) if anthropic_key else None

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)

    def disconnect(self, websocket: WebSocket):
        if websocket in self.active_connections:
            self.active_connections.remove(websocket)

    async def generate_tts(self, text: str, websocket: WebSocket):
        """Calls Deepgram TTS and sends binary audio chunks to the WebSocket."""
        if not self.dg_key:
            return
            
        try:
            # We use an async HTTP client for the Deepgram REST TTS API
            # so we don't block the event loop
            async with httpx.AsyncClient() as client:
                response = await client.post(
                    "https://api.deepgram.com/v1/speak?model=aura-asteria-en",
                    headers={
                        "Authorization": f"Token {self.dg_key}",
                        "Content-Type": "application/json"
                    },
                    json={"text": text},
                    timeout=5.0
                )
                if response.status_code == 200:
                    # Send audio bytes directly to the client
                    await websocket.send_bytes(response.content)
                else:
                    print(f"Deepgram TTS Error: {response.text}")
        except asyncio.CancelledError:
            raise
        except Exception as e:
            print(f"TTS Error: {e}")

    async def run_rag_pipeline(self, query: str, websocket: WebSocket):
        """Runs the RAG retrieval, LLM generation, and streams TTS back."""
        try:
            # 1. Retrieval
            retrieval_start = time.time()
            search_res = await asyncio.to_thread(self.vdb_manager.search, query, top_k=3)
            retrieval_latency = (time.time() - retrieval_start) * 1000
            sources = search_res["results"].get("matches", [])
            
            # Send sources to UI
            await websocket.send_text(json.dumps({'type': 'sources', 'sources': [{'text': s['metadata']['text'], 'strategy': s['metadata']['strategy'], 'score': s.get('score', 0)} for s in sources]}))

            context_str = "\n\n".join([f"Source [{i+1}] (Match Score: {s.get('score', 0):.1%}, Strategy: {s['metadata']['strategy']}):\n{s['metadata']['text']}" for i, s in enumerate(sources)])
            system_prompt = "You are a helpful, extremely concise AI voice assistant. Keep answers short (1-3 sentences maximum) so they can be spoken quickly. Do NOT use markdown, asterisks, or formatting in your response. Just plain text."
            user_prompt = f"Context:\n{context_str}\n\nQuestion: {query}\n\nConcise Answer:"

            if not self.groq_client and not self.anthropic_client:
                await websocket.send_text(json.dumps({'type': 'content', 'content': "No LLM API keys configured."}))
                return

            # 2. LLM Generation
            buffer = ""
            sentence_buffer = ""
            
            # Using Groq Llama 3 for ultra-low latency voice responses
            if self.groq_client:
                completion = self.groq_client.chat.completions.create(
                    model="qwen/qwen3.6-27b",
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt}
                    ],
                    temperature=0.3,
                    max_tokens=100,
                    stream=True,
                    timeout=5.0
                )
                
                for chunk in completion:
                    if chunk.choices[0].delta.content is not None:
                        token = chunk.choices[0].delta.content
                        
                        # Send text token to UI for live display
                        await websocket.send_text(json.dumps({'type': 'content', 'content': token}))
                        
                        sentence_buffer += token
                        
                        # If we hit a sentence boundary, trigger TTS
                        if any(p in token for p in [".", "?", "!", "\n"]):
                            clean_sentence = sentence_buffer.strip().replace("*", "")
                            if len(clean_sentence) > 2:
                                # Fire and forget the TTS task for this sentence
                                asyncio.create_task(self.generate_tts(clean_sentence, websocket))
                            sentence_buffer = ""
                            
                # Flush remaining buffer
                if len(sentence_buffer.strip()) > 2:
                    clean_sentence = sentence_buffer.strip().replace("*", "")
                    asyncio.create_task(self.generate_tts(clean_sentence, websocket))

            await websocket.send_text(json.dumps({'type': 'done'}))

        except asyncio.CancelledError:
            print("RAG Pipeline cancelled due to user interruption (Barge-in).")
            raise
        except Exception as e:
            print(f"RAG Error: {e}")

    async def handle_voice_session(self, websocket: WebSocket):
        await self.connect(websocket)
        
        if not self.deepgram:
            await websocket.send_text(json.dumps({"type": "error", "message": "Deepgram API key not configured on server."}))
            await websocket.close()
            return
            
        dg_connection = self.deepgram.listen.asyncwebsocket.v("1")
        llm_task: asyncio.Task = None
        
        async def on_message(self, result, **kwargs):
            nonlocal llm_task
            sentence = result.channel.alternatives[0].transcript
            if len(sentence) == 0:
                return
            
            # If we receive ANY partial speech from the user, cancel the active LLM/TTS immediately (Barge-in!)
            if llm_task and not llm_task.done():
                print("Barge-in detected! Cancelling current response...")
                llm_task.cancel()
            
            if result.is_final:
                print(f"Final User Utterance: {sentence}")
                await websocket.send_text(json.dumps({"type": "transcript", "text": sentence}))
                
                # Start the RAG pipeline asynchronously so it doesn't block the WebSocket read loop
                llm_task = asyncio.create_task(self.run_rag_pipeline(sentence, websocket))
                
        dg_connection.on(LiveTranscriptionEvents.Transcript, on_message)
        
        options = LiveOptions(
            model="nova-2",
            language="en-US",
            smart_format=True,
            endpointing=300 # 300ms of silence triggers a final utterance
        )
        
        if await dg_connection.start(options) is False:
            print("Failed to connect to Deepgram STT")
            return

        try:
            while True:
                data = await websocket.receive()
                if "bytes" in data:
                    # Route binary mic audio to Deepgram STT
                    await dg_connection.send(data["bytes"])
                elif "text" in data:
                    msg = json.loads(data["text"])
                    if msg.get("type") == "interrupt":
                        print("User interrupted manually.")
                        if llm_task and not llm_task.done():
                            llm_task.cancel()
        except WebSocketDisconnect:
            print("Client disconnected.")
            self.disconnect(websocket)
        except Exception as e:
            print(f"WS Error: {e}")
            self.disconnect(websocket)
        finally:
            if llm_task and not llm_task.done():
                llm_task.cancel()
            await dg_connection.finish()
