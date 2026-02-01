"use client";

import { useState, useRef, useCallback } from "react";
import { voiceQuery, getVoiceSpeakUrl } from "@/lib/api";

interface Message {
  role: "user" | "assistant";
  text: string;
  timestamp: Date;
}

type VoiceState = "idle" | "listening" | "processing" | "speaking";

export default function VoiceChat() {
  const [state, setState] = useState<VoiceState>("idle");
  const [messages, setMessages] = useState<Message[]>([]);
  const [error, setError] = useState<string | null>(null);
  
  const mediaRecorderRef = useRef<MediaRecorder | null>(null);
  const audioChunksRef = useRef<Blob[]>([]);
  const audioRef = useRef<HTMLAudioElement | null>(null);

  const startRecording = useCallback(async () => {
    try {
      setError(null);
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
      const mediaRecorder = new MediaRecorder(stream);
      mediaRecorderRef.current = mediaRecorder;
      audioChunksRef.current = [];

      mediaRecorder.ondataavailable = (event) => {
        audioChunksRef.current.push(event.data);
      };

      mediaRecorder.onstop = async () => {
        const audioBlob = new Blob(audioChunksRef.current, { type: "audio/webm" });
        stream.getTracks().forEach((track) => track.stop());
        
        // Process the audio
        setState("processing");
        
        try {
          const result = await voiceQuery(audioBlob);
          
          // Add user message
          setMessages((prev) => [
            ...prev,
            { role: "user", text: result.transcription, timestamp: new Date() },
          ]);
          
          // Add assistant message
          setMessages((prev) => [
            ...prev,
            { role: "assistant", text: result.response, timestamp: new Date() },
          ]);
          
          // Play audio response
          setState("speaking");
          const audio = new Audio(getVoiceSpeakUrl(result.response));
          audioRef.current = audio;
          
          audio.onended = () => {
            setState("idle");
          };
          
          audio.onerror = () => {
            setState("idle");
            setError("Failed to play audio response");
          };
          
          audio.play();
        } catch (e) {
          setError((e as Error).message);
          setState("idle");
        }
      };

      mediaRecorder.start();
      setState("listening");
    } catch (e) {
      setError("Microphone access denied");
      setState("idle");
    }
  }, []);

  const stopRecording = useCallback(() => {
    if (mediaRecorderRef.current && state === "listening") {
      mediaRecorderRef.current.stop();
    }
  }, [state]);

  const toggleRecording = useCallback(() => {
    if (state === "idle") {
      startRecording();
    } else if (state === "listening") {
      stopRecording();
    } else if (state === "speaking" && audioRef.current) {
      audioRef.current.pause();
      setState("idle");
    }
  }, [state, startRecording, stopRecording]);

  const getOrbStyle = () => {
    switch (state) {
      case "listening":
        return { background: "linear-gradient(135deg, #ef4444, #f97316)", animation: "pulse 1.5s ease-in-out infinite" };
      case "processing":
        return { background: "linear-gradient(135deg, #06b6d4, #0ea5e9)", animation: "spin 2s linear infinite" };
      case "speaking":
        return { background: "linear-gradient(135deg, #10b981, #34d399)", animation: "speak 0.5s ease-in-out infinite alternate" };
      default:
        return { background: "linear-gradient(135deg, #6366f1, #8b5cf6)" };
    }
  };

  const getStatusText = () => {
    switch (state) {
      case "listening":
        return "Listening... (tap to stop)";
      case "processing":
        return "Processing...";
      case "speaking":
        return "Speaking... (tap to stop)";
      default:
        return "Tap to speak";
    }
  };

  const getIcon = () => {
    switch (state) {
      case "listening":
        return "🔴";
      case "processing":
        return "⏳";
      case "speaking":
        return "🔊";
      default:
        return "🎤";
    }
  };

  return (
    <div className="voice-chat">
      <style jsx>{`
        .voice-chat {
          display: flex;
          flex-direction: column;
          align-items: center;
          padding: 2rem;
          gap: 1.5rem;
        }
        
        .orb-container {
          position: relative;
        }
        
        .orb {
          width: 120px;
          height: 120px;
          border-radius: 50%;
          display: flex;
          align-items: center;
          justify-content: center;
          cursor: pointer;
          font-size: 2.5rem;
          box-shadow: 0 0 40px rgba(99, 102, 241, 0.4);
          transition: transform 0.2s, box-shadow 0.2s;
        }
        
        .orb:hover {
          transform: scale(1.05);
          box-shadow: 0 0 60px rgba(99, 102, 241, 0.6);
        }
        
        .orb:active {
          transform: scale(0.95);
        }
        
        @keyframes pulse {
          0%, 100% { transform: scale(1); }
          50% { transform: scale(1.1); }
        }
        
        @keyframes spin {
          from { transform: rotate(0deg); }
          to { transform: rotate(360deg); }
        }
        
        @keyframes speak {
          0% { transform: scale(1); }
          100% { transform: scale(1.08); }
        }
        
        .status {
          font-size: 1rem;
          color: var(--text-dim);
        }
        
        .error {
          color: var(--red);
          font-size: 0.9rem;
        }
        
        .messages {
          width: 100%;
          max-width: 500px;
          max-height: 300px;
          overflow-y: auto;
          display: flex;
          flex-direction: column;
          gap: 0.75rem;
        }
        
        .message {
          padding: 0.75rem 1rem;
          border-radius: 12px;
          max-width: 85%;
        }
        
        .message.user {
          background: rgba(99, 102, 241, 0.2);
          align-self: flex-end;
          margin-left: auto;
        }
        
        .message.assistant {
          background: rgba(16, 185, 129, 0.2);
          align-self: flex-start;
        }
        
        .message-label {
          font-size: 0.7rem;
          color: var(--text-dim);
          margin-bottom: 0.25rem;
        }
        
        .hint {
          font-size: 0.85rem;
          color: var(--text-dim);
          text-align: center;
          margin-top: 1rem;
        }
      `}</style>
      
      <div className="orb-container">
        <div 
          className="orb" 
          style={getOrbStyle()}
          onClick={toggleRecording}
        >
          {getIcon()}
        </div>
      </div>
      
      <div className="status">{getStatusText()}</div>
      
      {error && <div className="error">{error}</div>}
      
      {messages.length > 0 && (
        <div className="messages">
          {messages.map((msg, i) => (
            <div key={i} className={`message ${msg.role}`}>
              <div className="message-label">
                {msg.role === "user" ? "You" : "K1"}
              </div>
              {msg.text}
            </div>
          ))}
        </div>
      )}
      
      <div className="hint">
        Ask: "What do you see?" • "Describe what's in front of you"
      </div>
    </div>
  );
}

