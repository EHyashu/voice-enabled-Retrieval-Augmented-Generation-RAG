import React, { useState, useRef, useEffect } from 'react';
import { Mic, Square, Play, RefreshCw, Layers, Database, Cpu, MessageSquare, Settings } from 'lucide-react';

interface Source {
  text: string;
  strategy: string;
  score: number;
}

interface LatencyMetrics {
  recordingLengthMs: number;
  sttLatencyMs: number;
  retrievalLatencyMs: number;
  llmFirstTokenLatencyMs: number;
  totalLatencyMs: number;
}

export const VoiceRAG: React.FC = () => {
  // Input & State
  const [isRecording, setIsRecording] = useState(false);
  const [transcription, setTranscription] = useState('');
  const [answer, setAnswer] = useState('');
  const [sources, setSources] = useState<Source[]>([]);
  const [loading, setLoading] = useState(false);
  const [recordingTime, setRecordingTime] = useState(0);
  const [textQuery, setTextQuery] = useState('');
  const [activeTab, setActiveTab] = useState<'voice' | 'text' | 'upload'>('voice');
  const [showSettings, setShowSettings] = useState(false);
  
  // File Upload State
  const [uploadFile, setUploadFile] = useState<File | null>(null);
  const [uploading, setUploading] = useState(false);
  const [uploadStatus, setUploadStatus] = useState('');
  // API Configuration Keys
  const [sarvamKey, setSarvamKey] = useState(() => localStorage.getItem('SARVAM_API_KEY') || '');
  const [backendUrl, setBackendUrl] = useState(() => localStorage.getItem('BACKEND_URL') || 'http://localhost:8000');

  // Latency Metrics
  const [metrics, setMetrics] = useState<LatencyMetrics>({
    recordingLengthMs: 0,
    sttLatencyMs: 0,
    retrievalLatencyMs: 0,
    llmFirstTokenLatencyMs: 0,
    totalLatencyMs: 0,
  });

  // Refs for Recording
  const recordingStartRef = useRef<number>(0);
  const timerRef = useRef<number | null>(null);
  const recognitionRef = useRef<any>(null);

  // Save Settings to Local Storage
  useEffect(() => {
    localStorage.setItem('SARVAM_API_KEY', sarvamKey);
  }, [sarvamKey]);

  useEffect(() => {
    localStorage.setItem('BACKEND_URL', backendUrl);
  }, [backendUrl]);

  // Audio  // Start Recording using Browser Native Speech Recognition
  const startRecording = async () => {
    // Prime speech synthesis to bypass browser autoplay restrictions
    window.speechSynthesis.cancel();
    window.speechSynthesis.speak(new SpeechSynthesisUtterance(""));
    
    setTranscription('');
    setAnswer('');
    setSources([]);
    setMetrics({
      recordingLengthMs: 0,
      sttLatencyMs: 0,
      retrievalLatencyMs: 0,
      llmFirstTokenLatencyMs: 0,
      totalLatencyMs: 0
    });

    const SpeechRecognition = (window as any).SpeechRecognition || (window as any).webkitSpeechRecognition;
    if (!SpeechRecognition) {
      alert("Native Speech Recognition is not supported in this browser. Please use Google Chrome or Microsoft Edge.");
      return;
    }

    try {
      const recognition = new SpeechRecognition();
      // Set to English to enforce English transcription
      recognition.lang = 'en-IN'; 
      recognition.continuous = true;
      recognition.interimResults = true;
      
      recognitionRef.current = recognition;
      
      let silenceTimer: number | null = null;
      
      recognition.onstart = () => {
        recordingStartRef.current = Date.now();
        setRecordingTime(0);
        setIsRecording(true);
        // Start a visual recording timer
        timerRef.current = window.setInterval(() => {
          setRecordingTime(prev => prev + 1);
        }, 1000);
      };

      recognition.onresult = async (event: any) => {
        let text = "";
        for (let i = 0; i < event.results.length; i++) {
          text += event.results[i][0].transcript;
        }
        
        const recordingLength = Date.now() - recordingStartRef.current;
        setMetrics(m => ({ ...m, recordingLengthMs: recordingLength, sttLatencyMs: 0 })); // 0ms STT Latency!
        setTranscription(text);
        
        // Voice Activity Detection (VAD) - wait for silence before submitting
        if (silenceTimer) clearTimeout(silenceTimer);
        
        silenceTimer = window.setTimeout(async () => {
          recognition.stop();
          if (text.trim() === '') {
            setAnswer("Could not hear anything. Please try speaking clearly or typing your query.");
            return;
          }
          
          setLoading(true);
          await queryBackendRAG(text, 0);
        }, 1200); // Wait 1.2 seconds of silence before auto-submitting
      };

      recognition.onerror = (event: any) => {
        console.error("Speech recognition error:", event.error);
        if (event.error !== 'aborted') {
          setTranscription("[Speech recognition failed]");
          setAnswer(`Microphone error: ${event.error}. Please try again or type your query manually.`);
        }
      };

      recognition.onend = () => {
        setIsRecording(false);
        if (timerRef.current) {
          clearInterval(timerRef.current);
          timerRef.current = null;
        }
        if (silenceTimer) {
          clearTimeout(silenceTimer);
        }
      };

      recognition.start();
    } catch (err) {
      console.error("Error starting speech recognition:", err);
      alert("Failed to access microphone. Please ensure permissions are granted.");
    }
  };

  const stopRecording = () => {
    if (recognitionRef.current && isRecording) {
      recognitionRef.current.stop();
    }
  };

  // (Deprecated) Process Audio via Sarvam STT - Removed to eliminate network latency

  // Submit Text Query manually
  const submitTextQuery = async () => {
    if (!textQuery.trim()) return;
    
    // Prime speech synthesis to bypass browser autoplay restrictions
    window.speechSynthesis.cancel();
    window.speechSynthesis.speak(new SpeechSynthesisUtterance(""));
    
    setTranscription(textQuery);
    setAnswer('');
    setSources([]);
    setLoading(true);
    setMetrics({
      recordingLengthMs: 0,
      sttLatencyMs: 0,
      retrievalLatencyMs: 0,
      llmFirstTokenLatencyMs: 0,
      totalLatencyMs: 0
    });
    
    await queryBackendRAG(textQuery, 0);
  };

  // Call FastAPI backend /api/rag endpoint and stream response
  // Speak out LLM response using Instant Browser Native TTS API
  const playVoiceOutput = (textToSpeak: string) => {
    // Clean text by removing markdown artifacts for cleaner pronunciation
    const cleanedText = textToSpeak
      .replace(/[\*\#\_]/g, '')
      .replace(/\[LLM Error:[^\]]*\]/g, '')
      .trim();
      
    if (!cleanedText) return;
    
    try {
      console.log("Playing voice output for:", cleanedText);
      
      const utterance = new SpeechSynthesisUtterance(cleanedText);
      (window as any)._currentUtterance = utterance; // Prevent Garbage Collection (Crucial bug in Chrome/Safari)
      
      // Select an actual voice object (fixes silent failure on many Chrome/Edge versions)
      const voices = window.speechSynthesis.getVoices();
      if (voices.length > 0) {
        const preferredVoice = voices.find(v => v.lang.startsWith('en')) || voices[0];
        utterance.voice = preferredVoice;
        utterance.lang = preferredVoice.lang;
      } else {
        utterance.lang = "en-US";
      }
      
      // Let the browser automatically pick the best native voice for the detected language
      window.speechSynthesis.speak(utterance);
      
      utterance.onstart = () => console.warn("✅ TTS successfully started speaking.");
      utterance.onerror = (e) => console.error("❌ TTS failed to play:", e);
    } catch (err) {
      console.error("Failed to play Native TTS voice output:", err);
    }
  };

  const stopVoiceOutput = () => {
    window.speechSynthesis.cancel();
  };

  // Call FastAPI backend /api/rag endpoint and stream response
  const queryBackendRAG = async (queryText: string, sttLatency: number) => {
    setAnswer('');
    const backendStart = timeNow();
    let accumulatedAnswer = "";
    let spokenLength = 0;
    
    try {
      const response = await fetch(`${backendUrl}/api/rag`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json'
        },
        body: JSON.stringify({ query: queryText })
      });
      
      if (!response.ok) {
        throw new Error(`Backend request failed with status: ${response.status}`);
      }
      
      const reader = response.body?.getReader();
      const decoder = new TextDecoder();
      let buffer = "";
      
      if (!reader) {
        throw new Error("Failed to read stream from backend");
      }
      
      while (true) {
        const { value, done } = await reader.read();
        if (done) break;
        
        buffer += decoder.decode(value, { stream: true });
        const lines = buffer.split('\n\n');
        buffer = lines.pop() || "";
        
        for (const line of lines) {
          const cleanLine = line.trim();
          if (cleanLine.startsWith('data: ')) {
            try {
              const jsonStr = cleanLine.substring(6);
              const payload = JSON.parse(jsonStr);
              
              if (payload.type === 'sources') {
                setSources(payload.sources || []);
              } else if (payload.type === 'metrics') {
                const totalBack = timeNow() - backendStart;
                setMetrics(prev => ({
                  ...prev,
                  retrievalLatencyMs: payload.retrieval_latency || 0,
                  llmFirstTokenLatencyMs: payload.first_token_latency || 0,
                  totalLatencyMs: sttLatency + totalBack
                }));
              } else if (payload.type === 'content') {
                accumulatedAnswer += payload.content;
                setAnswer(prev => prev + payload.content);
                
                // Sentence-by-Sentence TTS Streaming
                const unspoken = accumulatedAnswer.substring(spokenLength);
                const sentenceMatch = unspoken.match(/[^.!?।]+[.!?।]+/);
                if (sentenceMatch) {
                  const sentence = sentenceMatch[0];
                  playVoiceOutput(sentence);
                  spokenLength += unspoken.indexOf(sentence) + sentence.length;
                }
              } else if (payload.type === 'done') {
                setLoading(false);
                // Speak any remaining leftover text
                const remainingText = accumulatedAnswer.substring(spokenLength);
                if (remainingText.trim()) {
                  playVoiceOutput(remainingText);
                }
              } else if (payload.type === 'error') {
                setAnswer(prev => prev + `\n[LLM Error: ${payload.error}]`);
                setLoading(false);
              }
            } catch (err) {
              console.error("SSE parse error:", cleanLine, err);
            }
          }
        }
      }
      
    } catch (error: any) {
      console.error("RAG Backend Error:", error);
      setAnswer(`Backend connection error: ${error.message}. Is your FastAPI backend running at ${backendUrl}?`);
      setLoading(false);
    }
  };

  const timeNow = () => performance.now();

  const formatTime = (seconds: number) => {
    const mins = Math.floor(seconds / 60);
    const secs = seconds % 60;
    return `${mins}:${secs < 10 ? '0' : ''}${secs}`;
  };

  const totalEndToEnd = metrics.sttLatencyMs + (metrics.totalLatencyMs ? (metrics.totalLatencyMs - metrics.sttLatencyMs) : 0);

  return (
    <div className="app-wrapper voice-rag-container">
      {/* Background Floating Doodles from Blush Design Inspiration */}
      <div className="bg-doodles">
        <div className="doodle doodle-circle-large"></div>
        <div className="doodle doodle-triangle-yellow"></div>
        <div className="doodle doodle-circle-pink"></div>
        <div className="doodle doodle-circle-small-cyan"></div>
        <div className="doodle doodle-star-cyan"></div>
        <svg className="doodle doodle-zigzag" viewBox="0 0 100 20" preserveAspectRatio="none">
          <path d="M0,10 L10,0 L20,10 L30,0 L40,10 L50,0 L60,10 L70,0 L80,10 L90,0 L100,10" fill="none" stroke="var(--accent-cyan)" strokeWidth="3" />
        </svg>
      </div>

      {/* Header */}
      <div className="header-section">
        <div className="brand-badge animate-pulse">HH GOA 2026</div>
        <h1>Voice RAG Orchestrator</h1>
        <p className="subtitle">Real-time Multilingual Audio Retrieval-Augmented Generation</p>
        
        <button className="settings-btn" onClick={() => setShowSettings(!showSettings)}>
          <Settings size={18} />
          <span>Config</span>
        </button>
      </div>

      {/* Settings Panel */}
      {showSettings && (
        <div className="settings-panel card-glass slide-down">
          <h3>Configuration Keys</h3>
          <div className="settings-grid">
            <div className="form-group">
              <label>Sarvam AI API Subscription Key (STT):</label>
              <input 
                type="password" 
                placeholder="Paste your api-subscription-key" 
                value={sarvamKey} 
                onChange={(e) => setSarvamKey(e.target.value)} 
              />
              <span className="help-text">Leave blank to use mock STT fallback for testing.</span>
            </div>
            <div className="form-group">
              <label>FastAPI Backend URL:</label>
              <input 
                type="text" 
                value={backendUrl} 
                onChange={(e) => setBackendUrl(e.target.value)} 
              />
            </div>
          </div>
        </div>
      )}

      {/* Main Console Layout */}
      <div className="console-grid">
        {/* Interaction Card */}
        <div className="card-glass console-left">
          <div className="tab-header">
            <button 
              className={`tab-btn ${activeTab === 'voice' ? 'active' : ''}`}
              onClick={() => setActiveTab('voice')}
            >
              <Mic size={16} />
              <span>Voice Capture</span>
            </button>
            <button 
              className={`tab-btn ${activeTab === 'text' ? 'active' : ''}`}
              onClick={() => setActiveTab('text')}
            >
              <MessageSquare size={16} />
              <span>Text Query</span>
            </button>
            <button 
              className={`tab-btn ${activeTab === 'upload' ? 'active' : ''}`}
              onClick={() => setActiveTab('upload')}
            >
              <Layers size={16} />
              <span>Upload Doc</span>
            </button>
          </div>

          <div className="tab-content">
            {activeTab === 'voice' ? (
              <div className="voice-tab">
                <div className={`mic-container ${isRecording ? 'recording' : ''}`}>
                  <div className="pulse-ring ring-1"></div>
                  <div className="pulse-ring ring-2"></div>
                  <button 
                    onClick={isRecording ? stopRecording : startRecording}
                    className={`mic-button ${isRecording ? 'active' : ''}`}
                    disabled={loading}
                  >
                    {isRecording ? <Square size={28} /> : <Mic size={32} />}
                  </button>
                </div>
                
                <div className="status-label">
                  {isRecording ? (
                    <span className="text-recording animate-pulse">
                      Recording... {formatTime(recordingTime)}
                    </span>
                  ) : loading ? (
                    <span className="text-loading">
                      <RefreshCw className="spin" size={14} /> Processing pipeline...
                    </span>
                  ) : (
                    <span className="text-ready">Tap to speak (supports Hindi & English)</span>
                  )}
                </div>
              </div>
            ) : activeTab === 'text' ? (
              <div className="text-tab">
                <textarea
                  placeholder="Type your question here (e.g. भारत के प्रधानमंत्री कौन हैं?)..."
                  value={textQuery}
                  onChange={(e) => setTextQuery(e.target.value)}
                  onKeyDown={(e) => {
                    if (e.key === 'Enter' && !e.shiftKey) {
                      e.preventDefault();
                      submitTextQuery();
                    }
                  }}
                  disabled={loading}
                />
                <button 
                  onClick={submitTextQuery}
                  className="submit-btn"
                  disabled={loading || !textQuery.trim()}
                >
                  {loading ? <RefreshCw className="spin" size={16} /> : <Play size={16} />}
                  <span>Submit Question</span>
                </button>
              </div>
            ) : activeTab === 'upload' ? (
              <div className="upload-tab" style={{ textAlign: 'center', padding: '2rem 1rem' }}>
                <div style={{ marginBottom: '1.5rem' }}>
                  <input
                    type="file"
                    accept=".pdf,.txt"
                    onChange={(e) => setUploadFile(e.target.files?.[0] || null)}
                    style={{
                      background: 'var(--bg-primary)',
                      color: 'white',
                      padding: '1rem',
                      borderRadius: '12px',
                      border: '3px solid var(--border-color)',
                      width: '100%',
                      fontFamily: "'Quicksand', sans-serif",
                      fontWeight: 700
                    }}
                  />
                </div>
                <button 
                  onClick={async () => {
                    if (!uploadFile) return;
                    setUploading(true);
                    setUploadStatus('Uploading and Chunking...');
                    try {
                      const formData = new FormData();
                      formData.append('file', uploadFile);
                      const response = await fetch(`${backendUrl}/api/upload`, {
                        method: 'POST',
                        body: formData
                      });
                      const data = await response.json();
                      if (response.ok) {
                        setUploadStatus(`Success! ${data.message}`);
                      } else {
                        setUploadStatus(`Error: ${data.detail}`);
                      }
                    } catch (e: any) {
                      setUploadStatus(`Upload failed: ${e.message}`);
                    } finally {
                      setUploading(false);
                      setUploadFile(null);
                    }
                  }}
                  className="submit-btn"
                  disabled={uploading || !uploadFile}
                  style={{ background: 'var(--accent-cyan)', color: 'var(--text-dark)' }}
                >
                  {uploading ? <RefreshCw className="spin" size={16} /> : <Database size={16} />}
                  <span>{uploading ? 'Processing...' : 'Ingest Document to Knowledge Base'}</span>
                </button>
                {uploadStatus && (
                  <p style={{ marginTop: '1rem', fontWeight: 800, color: 'var(--accent-pink)' }}>
                    {uploadStatus}
                  </p>
                )}
              </div>
            ) : null}
          </div>

          {/* Transcribed Text Result */}
          {transcription && (
            <div className="transcription-result">
              <span className="label">Transcribed Query:</span>
              <p className="transcription-text">"{transcription}"</p>
            </div>
          )}
        </div>

        {/* Answer Dashboard */}
        <div className="card-glass console-right">
          <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
            <h3>Generated Answer</h3>
            <button 
              onClick={stopVoiceOutput}
              style={{
                background: 'var(--accent-pink)',
                border: '2px solid var(--border-color)',
                color: 'var(--text-main)',
                padding: '8px 16px',
                borderRadius: '9999px',
                cursor: 'pointer',
                fontSize: '12px',
                fontWeight: '800',
                boxShadow: 'var(--shadow-solid)'
              }}
            >
              Stop Audio
            </button>
          </div>
          <div className="answer-content">
            {answer ? (
              <p className="answer-text">{answer}</p>
            ) : loading ? (
              <div className="answer-loader">
                <div className="loader-dots">
                  <span></span><span></span><span></span>
                </div>
                <p>Synthesizing context & generating response...</p>
              </div>
            ) : (
              <p className="answer-placeholder">Output answer will stream here...</p>
            )}
          </div>

          {/* Sources Section */}
          {sources.length > 0 && (
            <details className="sources-section blush-accordion">
              <summary>
                <Layers size={14} />
                <span>Retrieved Context ({sources.length} sources)</span>
              </summary>
              <div className="sources-list">
                {sources.map((src, idx) => (
                  <div key={idx} className="source-card">
                    <div className="source-header">
                      <span className={`strategy-badge ${src.strategy}`}>
                        {src.strategy}
                      </span>
                      {src.score > 0 && (
                        <span className="score-badge">
                          Match: {(src.score * 100).toFixed(1)}%
                        </span>
                      )}
                    </div>
                    <p className="source-text">{src.text}</p>
                  </div>
                ))}
              </div>
            </details>
          )}
        </div>
      </div>

      {/* Latency Dashboard */}
      <details className="card-glass latency-dashboard blush-accordion">
        <summary className="latency-header">
          <h3>Latency Budget Breakdown</h3>
          <div className={`latency-status ${totalEndToEnd < 200 && totalEndToEnd > 0 ? 'budget-ok' : totalEndToEnd >= 200 ? 'budget-exceeded' : ''}`}>
            {totalEndToEnd > 0 ? (
              <span>Total Backend + STT: {totalEndToEnd.toFixed(0)}ms</span>
            ) : (
              <span>Target: &lt; 200ms end-to-end</span>
            )}
          </div>
        </summary>

        <div className="latency-timeline">
          {/* Audio Capturing */}
          <div className="timeline-node">
            <div className="node-icon"><Mic size={14} /></div>
            <div className="node-info">
              <span className="node-name">Audio Capture</span>
              <span className="node-val">
                {metrics.recordingLengthMs > 0 ? `${(metrics.recordingLengthMs / 1000).toFixed(1)}s` : '0ms'}
              </span>
            </div>
          </div>

          {/* STT Transcription */}
          <div className="timeline-node">
            <div className="node-icon"><Database size={14} /></div>
            <div className="node-info">
              <span className="node-name">Sarvam STT</span>
              <span className="node-val">
                {metrics.sttLatencyMs > 0 ? `${metrics.sttLatencyMs.toFixed(0)}ms` : '0ms'}
              </span>
            </div>
            <div className="progress-bar">
              <div 
                className="progress-fill fill-stt" 
                style={{ width: `${Math.min(100, (metrics.sttLatencyMs / 200) * 100)}%` }}
              ></div>
            </div>
          </div>

          {/* Vector Retrieval */}
          <div className="timeline-node">
            <div className="node-icon"><Cpu size={14} /></div>
            <div className="node-info">
              <span className="node-name">Vector Search</span>
              <span className="node-val">
                {metrics.retrievalLatencyMs > 0 ? `${metrics.retrievalLatencyMs.toFixed(1)}ms` : '0ms'}
              </span>
            </div>
            <div className="progress-bar">
              <div 
                className="progress-fill fill-retrieval" 
                style={{ width: `${Math.min(100, (metrics.retrievalLatencyMs / 50) * 100)}%` }}
              ></div>
            </div>
          </div>

          {/* LLM Generation */}
          <div className="timeline-node">
            <div className="node-icon"><MessageSquare size={14} /></div>
            <div className="node-info">
              <span className="node-name">Claude LLM (TTFT)</span>
              <span className="node-val">
                {metrics.llmFirstTokenLatencyMs > 0 ? `${metrics.llmFirstTokenLatencyMs.toFixed(0)}ms` : '0ms'}
              </span>
            </div>
            <div className="progress-bar">
              <div 
                className="progress-fill fill-llm" 
                style={{ width: `${Math.min(100, (metrics.llmFirstTokenLatencyMs / 80) * 100)}%` }}
              ></div>
            </div>
          </div>
        </div>

        <div className="budget-comparison">
          <div className="budget-bar">
            <div className="budget-fill target" style={{ width: '100%' }}>
              <span className="budget-label">200ms Budget</span>
            </div>
            {totalEndToEnd > 0 && (
              <div 
                className={`budget-fill actual ${totalEndToEnd > 200 ? 'over' : ''}`}
                style={{ width: `${Math.min(100, (totalEndToEnd / 200) * 100)}%` }}
              >
                <span className="budget-label">Actual: {totalEndToEnd.toFixed(0)}ms</span>
              </div>
            )}
          </div>
        </div>
      </details>
    </div>
  );
};
