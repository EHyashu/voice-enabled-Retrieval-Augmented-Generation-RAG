import React, { useState, useRef, useEffect } from 'react';
import { Mic, Square, Play, Database, MessageSquare } from 'lucide-react';

interface Source {
  text: string;
  strategy: string;
  score: number;
}

export const VoiceRAG: React.FC = () => {
  const [isRecording, setIsRecording] = useState(false);
  const [transcription, setTranscription] = useState('');
  const [answer, setAnswer] = useState('');
  const [sources, setSources] = useState<Source[]>([]);
  const [loading, setLoading] = useState(false);
  const [recordingTime, setRecordingTime] = useState(0);
  
  // Audio Context & WebSocket
  const wsRef = useRef<WebSocket | null>(null);
  const mediaRecorderRef = useRef<MediaRecorder | null>(null);
  const audioContextRef = useRef<AudioContext | null>(null);
  const nextPlayTimeRef = useRef<number>(0);
  const timerRef = useRef<number | null>(null);

  // Initialize Audio Context on demand (needs user interaction)
  const initAudio = () => {
    if (!audioContextRef.current) {
      const AudioCtx = (window.AudioContext || (window as any).webkitAudioContext) as typeof AudioContext;
      audioContextRef.current = new AudioCtx();
    }
  };

  const connectWebSocket = () => {
    if (wsRef.current?.readyState === WebSocket.OPEN) return;
    
    // Connect to FastAPI WebSocket backend
    const ws = new WebSocket(`ws://${window.location.host}/api/ws/voice`);
    ws.binaryType = 'blob';
    
    ws.onmessage = async (event) => {
      if (typeof event.data === 'string') {
        const msg = JSON.parse(event.data);
        if (msg.type === 'transcript') {
          setTranscription(msg.text);
          setLoading(true);
        } else if (msg.type === 'sources') {
          setSources(msg.sources);
        } else if (msg.type === 'content') {
          setAnswer(prev => prev + msg.content);
          setLoading(false);
        } else if (msg.type === 'done') {
          setLoading(false);
        }
      } else if (event.data instanceof Blob) {
        // Play binary TTS audio chunks
        initAudio();
        const arrayBuffer = await event.data.arrayBuffer();
        const ctx = audioContextRef.current!;
        ctx.decodeAudioData(arrayBuffer, (buffer) => {
          const source = ctx.createBufferSource();
          source.buffer = buffer;
          source.connect(ctx.destination);
          
          if (nextPlayTimeRef.current < ctx.currentTime) {
            nextPlayTimeRef.current = ctx.currentTime;
          }
          source.start(nextPlayTimeRef.current);
          nextPlayTimeRef.current += buffer.duration;
        });
      }
    };
    
    ws.onerror = (err) => console.error("WS Error", err);
    wsRef.current = ws;
  };

  useEffect(() => {
    connectWebSocket();
    return () => {
      wsRef.current?.close();
    };
  }, []);

  const startRecording = async () => {
    initAudio();
    if (audioContextRef.current?.state === 'suspended') {
      audioContextRef.current.resume();
    }
    
    setTranscription('');
    setAnswer('');
    setSources([]);
    
    // Interrupt existing generation/audio
    if (wsRef.current?.readyState === WebSocket.OPEN) {
      wsRef.current.send(JSON.stringify({ type: 'interrupt' }));
    }
    nextPlayTimeRef.current = 0; // reset audio queue
    
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
      const mediaRecorder = new MediaRecorder(stream, { mimeType: 'audio/webm' });
      
      mediaRecorder.ondataavailable = (event) => {
        if (event.data.size > 0 && wsRef.current?.readyState === WebSocket.OPEN) {
          wsRef.current.send(event.data);
        }
      };
      
      mediaRecorder.start(250); // send chunks every 250ms
      mediaRecorderRef.current = mediaRecorder;
      setIsRecording(true);
      
      timerRef.current = window.setInterval(() => {
        setRecordingTime(prev => prev + 1);
      }, 1000);
      
    } catch (err) {
      console.error("Mic access denied", err);
      alert("Microphone access is required.");
    }
  };

  const stopRecording = () => {
    if (mediaRecorderRef.current && isRecording) {
      mediaRecorderRef.current.stop();
      mediaRecorderRef.current.stream.getTracks().forEach(track => track.stop());
    }
    setIsRecording(false);
    if (timerRef.current) clearInterval(timerRef.current);
    setRecordingTime(0);
  };

  return (
    <div className="max-w-6xl mx-auto flex flex-col md:flex-row gap-6 p-6">
      <div className="flex-1 space-y-6">
        <div className="bg-white rounded-3xl p-8 shadow-sm border border-neutral-100 flex flex-col items-center justify-center text-center">
          <div className="mb-6 relative">
             <button
               onClick={isRecording ? stopRecording : startRecording}
               className={`relative z-10 w-24 h-24 rounded-full flex items-center justify-center transition-all duration-500 shadow-xl ${
                 isRecording 
                   ? 'bg-red-500 text-white hover:bg-red-600 hover:scale-95' 
                   : 'bg-black text-white hover:bg-gray-900 hover:scale-105'
               }`}
             >
               {isRecording ? <Square size={32} fill="currentColor" /> : <Mic size={36} />}
             </button>
             {isRecording && (
                <div className="absolute inset-0 bg-red-500 rounded-full animate-ping opacity-30 z-0"></div>
             )}
          </div>
          <h2 className="text-2xl font-bold tracking-tight text-neutral-900 mb-2">
            {isRecording ? "Listening..." : "Tap to Speak"}
          </h2>
          {isRecording && (
            <div className="text-red-500 font-medium">
              {Math.floor(recordingTime / 60).toString().padStart(2, '0')}:{(recordingTime % 60).toString().padStart(2, '0')}
            </div>
          )}
        </div>

        <div className="bg-white rounded-3xl p-8 shadow-sm border border-neutral-100">
          <h3 className="text-lg font-bold flex items-center gap-2 mb-4"><MessageSquare size={20} /> Transcription</h3>
          <p className="text-neutral-700 min-h-[4rem] p-4 bg-neutral-50 rounded-2xl italic">
            {transcription || "Your transcription will appear here..."}
          </p>
        </div>

        <div className="bg-white rounded-3xl p-8 shadow-sm border border-neutral-100 min-h-[250px]">
          <h3 className="text-lg font-bold flex items-center gap-2 mb-4 text-blue-600"><Play size={20} /> AI Response</h3>
          <p className="text-neutral-800 text-lg leading-relaxed whitespace-pre-wrap">
            {answer || (loading ? "Generating voice response..." : "AI voice response will appear here as it plays...")}
          </p>
        </div>
      </div>
      
      <div className="w-full md:w-96 space-y-6">
        <div className="bg-white rounded-3xl p-6 shadow-sm border border-neutral-100">
          <h3 className="font-bold flex items-center gap-2 mb-4 text-neutral-900">
             <Database size={18} className="text-orange-500" /> Vector Sources
          </h3>
          <div className="space-y-4">
            {sources.length === 0 ? (
              <p className="text-neutral-500 text-sm">No sources retrieved.</p>
            ) : (
              sources.map((s, i) => (
                <div key={i} className="p-4 bg-orange-50/50 rounded-2xl border border-orange-100">
                  <div className="flex justify-between items-center mb-2">
                    <span className="text-xs font-bold bg-orange-200 text-orange-800 px-2 py-1 rounded-full">{s.strategy}</span>
                    <span className="text-xs font-semibold text-orange-600">{(s.score * 100).toFixed(1)}% Match</span>
                  </div>
                  <p className="text-sm text-neutral-700 line-clamp-3">{s.text}</p>
                </div>
              ))
            )}
          </div>
        </div>
      </div>
    </div>
  );
};
