#!/usr/bin/env python3
"""
Voice Assistant for K1 Robot - Ask what it sees via voice
Uses Pipecat for real-time voice AI with vision capabilities

Supports two modes:
1. WebRTC mode (via Daily.co) - for browser-based voice interaction
2. Local mode - for direct microphone/speaker interaction
"""

import asyncio
import base64
import os
import io
import sys
import cv2
import numpy as np
from datetime import datetime
from typing import Optional
from dotenv import load_dotenv

# Pipecat imports
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.runner import PipelineRunner
from pipecat.pipeline.task import PipelineParams, PipelineTask
from pipecat.frames.frames import (
    Frame,
    TextFrame,
    ImageRawFrame,
    LLMMessagesFrame,
    EndFrame,
)
from pipecat.processors.frame_processor import FrameProcessor, FrameDirection
from pipecat.services.openai import OpenAILLMService, OpenAITTSService
from pipecat.services.deepgram import DeepgramSTTService, DeepgramTTSService
from pipecat.transports.services.daily import DailyParams, DailyTransport
from pipecat.transports.local.audio import LocalAudioTransport
from pipecat.vad.silero import SileroVADAnalyzer

load_dotenv()


class VisionFrameProvider:
    """Provides the latest camera frame for vision analysis."""
    
    def __init__(self):
        self.latest_frame: Optional[np.ndarray] = None
        self.latest_detections: list = []
        self.frame_lock = asyncio.Lock()
    
    async def update_frame(self, frame: np.ndarray, detections: list = None):
        """Update the latest frame and detections."""
        async with self.frame_lock:
            self.latest_frame = frame.copy()
            if detections:
                self.latest_detections = detections.copy()
    
    async def get_frame_base64(self) -> Optional[str]:
        """Get the latest frame as base64 encoded JPEG."""
        async with self.frame_lock:
            if self.latest_frame is None:
                return None
            
            # Encode frame to JPEG
            _, buffer = cv2.imencode('.jpg', self.latest_frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
            return base64.b64encode(buffer).decode('utf-8')
    
    async def get_detection_summary(self) -> str:
        """Get a text summary of current detections."""
        async with self.frame_lock:
            if not self.latest_detections:
                return "No objects currently detected."
            
            # Count objects by class
            counts = {}
            for det in self.latest_detections:
                cls = det.get('class_name', 'unknown')
                counts[cls] = counts.get(cls, 0) + 1
            
            parts = [f"{count} {name}{'s' if count > 1 else ''}" 
                    for name, count in counts.items()]
            return f"Currently detecting: {', '.join(parts)}"


class VisionQueryProcessor(FrameProcessor):
    """
    Processor that intercepts user queries about vision and adds image context.
    """
    
    def __init__(self, vision_provider: VisionFrameProvider, **kwargs):
        super().__init__(**kwargs)
        self.vision_provider = vision_provider
        self.vision_keywords = [
            "see", "look", "looking", "visible", "show", "camera",
            "what", "describe", "tell me", "is there", "are there",
            "front", "around", "detect", "object", "person", "thing"
        ]
    
    def _is_vision_query(self, text: str) -> bool:
        """Check if the query is about vision/what the bot sees."""
        text_lower = text.lower()
        return any(keyword in text_lower for keyword in self.vision_keywords)
    
    async def process_frame(self, frame: Frame, direction: FrameDirection):
        """Process frames and add vision context when needed."""
        await super().process_frame(frame, direction)
        
        if isinstance(frame, LLMMessagesFrame):
            # Check if any user message is a vision query
            messages = frame.messages
            
            for msg in messages:
                if msg.get("role") == "user":
                    content = msg.get("content", "")
                    if isinstance(content, str) and self._is_vision_query(content):
                        # Add image to the message
                        image_b64 = await self.vision_provider.get_frame_base64()
                        detection_summary = await self.vision_provider.get_detection_summary()
                        
                        if image_b64:
                            # Convert to multimodal message format
                            msg["content"] = [
                                {
                                    "type": "text",
                                    "text": f"[Current detections: {detection_summary}]\n\nUser question: {content}"
                                },
                                {
                                    "type": "image_url",
                                    "image_url": {
                                        "url": f"data:image/jpeg;base64,{image_b64}",
                                        "detail": "high"
                                    }
                                }
                            ]
                        else:
                            msg["content"] = f"[No camera feed available] User question: {content}"
        
        await self.push_frame(frame, direction)


class K1VoiceAssistant:
    """
    Voice assistant for K1 robot with vision capabilities.
    """
    
    def __init__(
        self,
        vision_provider: VisionFrameProvider,
        mode: str = "daily",  # "daily" or "local"
        daily_room_url: str = None,
        daily_token: str = None,
    ):
        self.vision_provider = vision_provider
        self.mode = mode
        self.daily_room_url = daily_room_url
        self.daily_token = daily_token
        
        # Validate API keys
        self._validate_env()
    
    def _validate_env(self):
        """Validate required environment variables."""
        required = ["OPENAI_API_KEY"]
        
        if self.mode == "daily":
            required.extend(["DAILY_API_KEY"])
        
        missing = [key for key in required if not os.getenv(key)]
        if missing:
            raise ValueError(f"Missing required environment variables: {', '.join(missing)}")
    
    def _get_system_prompt(self) -> str:
        """Get the system prompt for the assistant."""
        return """You are K1, a friendly robot assistant with vision capabilities. 
You can see through your camera and describe what you observe.

When the user asks about what you see:
1. Describe the scene naturally and conversationally
2. Mention specific objects you detect with their approximate positions (left, right, center, close, far)
3. If you detect people, describe their general activities or positions
4. Be helpful and informative but keep responses concise

If no image is provided or the camera feed is unavailable, let the user know politely.

You can also have general conversations, answer questions, and help the user with various tasks.
Keep your responses conversational and friendly - you're speaking out loud!"""
    
    async def create_pipeline(self) -> Pipeline:
        """Create the Pipecat pipeline."""
        
        # Create transport based on mode
        if self.mode == "daily":
            transport = DailyTransport(
                self.daily_room_url,
                self.daily_token,
                "K1 Robot",
                DailyParams(
                    audio_in_enabled=True,
                    audio_out_enabled=True,
                    video_out_enabled=False,
                    vad_enabled=True,
                    vad_analyzer=SileroVADAnalyzer(),
                    transcription_enabled=False,
                )
            )
        else:
            # Local audio transport
            transport = LocalAudioTransport(
                vad_analyzer=SileroVADAnalyzer()
            )
        
        # Speech-to-Text
        stt = DeepgramSTTService(
            api_key=os.getenv("DEEPGRAM_API_KEY", ""),
            # Fall back to OpenAI Whisper if no Deepgram key
        ) if os.getenv("DEEPGRAM_API_KEY") else None
        
        # LLM with vision capability (GPT-4o)
        llm = OpenAILLMService(
            api_key=os.getenv("OPENAI_API_KEY"),
            model="gpt-4o",
        )
        
        # Text-to-Speech
        tts = OpenAITTSService(
            api_key=os.getenv("OPENAI_API_KEY"),
            voice="nova",  # Options: alloy, echo, fable, onyx, nova, shimmer
        )
        
        # Vision query processor
        vision_processor = VisionQueryProcessor(self.vision_provider)
        
        # Create messages with system prompt
        messages = [
            {"role": "system", "content": self._get_system_prompt()}
        ]
        
        # Build pipeline
        pipeline_processors = [
            transport.input(),
        ]
        
        if stt:
            pipeline_processors.append(stt)
        
        pipeline_processors.extend([
            vision_processor,
            llm,
            tts,
            transport.output(),
        ])
        
        pipeline = Pipeline(pipeline_processors)
        
        return pipeline, transport, messages
    
    async def run(self):
        """Run the voice assistant."""
        pipeline, transport, messages = await self.create_pipeline()
        
        task = PipelineTask(
            pipeline,
            PipelineParams(
                allow_interruptions=True,
                enable_metrics=True,
            )
        )
        
        # Initialize with system message
        await task.queue_frames([LLMMessagesFrame(messages)])
        
        runner = PipelineRunner()
        await runner.run(task)


class SimplifiedVoiceBot:
    """
    Simplified voice bot using OpenAI's Realtime API for lower latency.
    This is an alternative implementation that doesn't require Daily.co.
    """
    
    def __init__(self, vision_provider: VisionFrameProvider):
        self.vision_provider = vision_provider
        self.running = False
    
    async def process_voice_command(self, audio_data: bytes) -> str:
        """Process voice command and return text response."""
        import openai
        
        client = openai.AsyncOpenAI()
        
        # Transcribe audio
        audio_file = io.BytesIO(audio_data)
        audio_file.name = "audio.wav"
        
        transcript = await client.audio.transcriptions.create(
            model="whisper-1",
            file=audio_file
        )
        
        user_text = transcript.text
        print(f"[User]: {user_text}")
        
        # Get vision context
        image_b64 = await self.vision_provider.get_frame_base64()
        detection_summary = await self.vision_provider.get_detection_summary()
        
        # Build message
        if image_b64:
            messages = [
                {
                    "role": "system",
                    "content": self._get_system_prompt()
                },
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": f"[{detection_summary}]\n\nUser: {user_text}"},
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{image_b64}",
                                "detail": "high"
                            }
                        }
                    ]
                }
            ]
        else:
            messages = [
                {"role": "system", "content": self._get_system_prompt()},
                {"role": "user", "content": user_text}
            ]
        
        # Get LLM response
        response = await client.chat.completions.create(
            model="gpt-4o",
            messages=messages,
            max_tokens=300
        )
        
        response_text = response.choices[0].message.content
        print(f"[K1]: {response_text}")
        
        return response_text
    
    async def text_to_speech(self, text: str) -> bytes:
        """Convert text to speech."""
        import openai
        
        client = openai.AsyncOpenAI()
        
        response = await client.audio.speech.create(
            model="tts-1",
            voice="nova",
            input=text
        )
        
        return response.content
    
    def _get_system_prompt(self) -> str:
        return """You are K1, a friendly robot assistant with vision capabilities.
You can see through your camera and describe what you observe.

When asked about what you see:
1. Describe the scene naturally and conversationally
2. Mention specific objects with their positions
3. Be helpful but keep responses concise (2-3 sentences)

Keep responses short and conversational - you're speaking out loud!"""


# HTTP API for web-based voice interaction
from fastapi import FastAPI, WebSocket, UploadFile, File, HTTPException
from fastapi.responses import StreamingResponse, HTMLResponse
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI(title="K1 Voice Assistant")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global vision provider - will be set by main script
vision_provider: Optional[VisionFrameProvider] = None
voice_bot: Optional[SimplifiedVoiceBot] = None


@app.get("/", response_class=HTMLResponse)
async def index():
    """Serve the voice assistant web interface."""
    return """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>K1 Voice Assistant</title>
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }
        
        body {
            font-family: 'SF Pro Display', -apple-system, BlinkMacSystemFont, sans-serif;
            background: linear-gradient(135deg, #0f0f23 0%, #1a1a3e 50%, #0d0d1a 100%);
            min-height: 100vh;
            display: flex;
            flex-direction: column;
            align-items: center;
            justify-content: center;
            color: #ffffff;
            overflow: hidden;
        }
        
        .container {
            text-align: center;
            padding: 2rem;
            max-width: 600px;
        }
        
        h1 {
            font-size: 3rem;
            margin-bottom: 0.5rem;
            background: linear-gradient(90deg, #00d4ff, #7c3aed, #f472b6);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            animation: gradient-shift 3s ease infinite;
        }
        
        @keyframes gradient-shift {
            0%, 100% { filter: hue-rotate(0deg); }
            50% { filter: hue-rotate(30deg); }
        }
        
        .subtitle {
            color: #888;
            margin-bottom: 3rem;
            font-size: 1.1rem;
        }
        
        .orb-container {
            position: relative;
            width: 200px;
            height: 200px;
            margin: 0 auto 2rem;
        }
        
        .orb {
            width: 100%;
            height: 100%;
            border-radius: 50%;
            background: radial-gradient(circle at 30% 30%, #4f46e5, #7c3aed, #1e1b4b);
            box-shadow: 
                0 0 60px rgba(124, 58, 237, 0.5),
                inset 0 0 60px rgba(255, 255, 255, 0.1);
            cursor: pointer;
            transition: all 0.3s ease;
            display: flex;
            align-items: center;
            justify-content: center;
        }
        
        .orb:hover {
            transform: scale(1.05);
            box-shadow: 
                0 0 80px rgba(124, 58, 237, 0.7),
                inset 0 0 60px rgba(255, 255, 255, 0.2);
        }
        
        .orb.listening {
            animation: pulse 1.5s ease-in-out infinite;
            background: radial-gradient(circle at 30% 30%, #ef4444, #f97316, #7c2d12);
            box-shadow: 
                0 0 80px rgba(239, 68, 68, 0.6),
                inset 0 0 60px rgba(255, 255, 255, 0.2);
        }
        
        .orb.processing {
            animation: spin 2s linear infinite;
            background: radial-gradient(circle at 30% 30%, #06b6d4, #0ea5e9, #0c4a6e);
        }
        
        .orb.speaking {
            animation: speak 0.5s ease-in-out infinite alternate;
            background: radial-gradient(circle at 30% 30%, #10b981, #34d399, #064e3b);
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
        
        .orb-icon {
            font-size: 3rem;
        }
        
        .status {
            font-size: 1.2rem;
            color: #a5b4fc;
            margin-bottom: 2rem;
            min-height: 2rem;
        }
        
        .transcript {
            background: rgba(255, 255, 255, 0.05);
            border: 1px solid rgba(255, 255, 255, 0.1);
            border-radius: 16px;
            padding: 1.5rem;
            margin-top: 2rem;
            max-height: 300px;
            overflow-y: auto;
            text-align: left;
        }
        
        .message {
            padding: 0.75rem 1rem;
            margin: 0.5rem 0;
            border-radius: 12px;
            animation: fadeIn 0.3s ease;
        }
        
        @keyframes fadeIn {
            from { opacity: 0; transform: translateY(10px); }
            to { opacity: 1; transform: translateY(0); }
        }
        
        .message.user {
            background: rgba(124, 58, 237, 0.3);
            margin-left: 2rem;
        }
        
        .message.assistant {
            background: rgba(16, 185, 129, 0.3);
            margin-right: 2rem;
        }
        
        .message-label {
            font-size: 0.75rem;
            color: #888;
            margin-bottom: 0.25rem;
        }
        
        .camera-preview {
            position: fixed;
            bottom: 20px;
            right: 20px;
            width: 200px;
            border-radius: 12px;
            overflow: hidden;
            border: 2px solid rgba(255, 255, 255, 0.2);
            box-shadow: 0 10px 40px rgba(0, 0, 0, 0.3);
        }
        
        .camera-preview img {
            width: 100%;
            height: auto;
            display: block;
        }
        
        .camera-label {
            position: absolute;
            bottom: 0;
            left: 0;
            right: 0;
            background: rgba(0, 0, 0, 0.7);
            padding: 0.5rem;
            font-size: 0.75rem;
            text-align: center;
        }
    </style>
</head>
<body>
    <div class="container">
        <h1>🤖 K1 Vision</h1>
        <p class="subtitle">Ask me what I see</p>
        
        <div class="orb-container">
            <div class="orb" id="orb" onclick="toggleRecording()">
                <span class="orb-icon" id="orbIcon">🎤</span>
            </div>
        </div>
        
        <div class="status" id="status">Tap to speak</div>
        
        <div class="transcript" id="transcript" style="display: none;"></div>
    </div>
    
    <div class="camera-preview">
        <img id="cameraFeed" src="/camera/latest" alt="Camera Feed">
        <div class="camera-label">📷 Live View</div>
    </div>
    
    <script>
        let mediaRecorder;
        let audioChunks = [];
        let isRecording = false;
        
        const orb = document.getElementById('orb');
        const orbIcon = document.getElementById('orbIcon');
        const status = document.getElementById('status');
        const transcript = document.getElementById('transcript');
        const cameraFeed = document.getElementById('cameraFeed');
        
        // Refresh camera feed
        setInterval(() => {
            cameraFeed.src = '/camera/latest?t=' + Date.now();
        }, 500);
        
        async function toggleRecording() {
            if (isRecording) {
                stopRecording();
            } else {
                startRecording();
            }
        }
        
        async function startRecording() {
            try {
                const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
                mediaRecorder = new MediaRecorder(stream);
                audioChunks = [];
                
                mediaRecorder.ondataavailable = (event) => {
                    audioChunks.push(event.data);
                };
                
                mediaRecorder.onstop = async () => {
                    const audioBlob = new Blob(audioChunks, { type: 'audio/webm' });
                    await sendAudio(audioBlob);
                };
                
                mediaRecorder.start();
                isRecording = true;
                
                orb.classList.add('listening');
                orbIcon.textContent = '🔴';
                status.textContent = 'Listening...';
                
            } catch (error) {
                console.error('Error accessing microphone:', error);
                status.textContent = 'Microphone access denied';
            }
        }
        
        function stopRecording() {
            if (mediaRecorder && isRecording) {
                mediaRecorder.stop();
                mediaRecorder.stream.getTracks().forEach(track => track.stop());
                isRecording = false;
                
                orb.classList.remove('listening');
                orb.classList.add('processing');
                orbIcon.textContent = '⏳';
                status.textContent = 'Processing...';
            }
        }
        
        async function sendAudio(audioBlob) {
            const formData = new FormData();
            formData.append('audio', audioBlob, 'recording.webm');
            
            try {
                const response = await fetch('/voice/query', {
                    method: 'POST',
                    body: formData
                });
                
                if (!response.ok) throw new Error('Server error');
                
                const data = await response.json();
                
                // Show transcript
                transcript.style.display = 'block';
                addMessage('You', data.transcription, 'user');
                addMessage('K1', data.response, 'assistant');
                
                // Play audio response
                orb.classList.remove('processing');
                orb.classList.add('speaking');
                orbIcon.textContent = '🔊';
                status.textContent = 'Speaking...';
                
                const audio = new Audio('/voice/speak?text=' + encodeURIComponent(data.response));
                audio.onended = () => {
                    orb.classList.remove('speaking');
                    orbIcon.textContent = '🎤';
                    status.textContent = 'Tap to speak';
                };
                audio.play();
                
            } catch (error) {
                console.error('Error:', error);
                orb.classList.remove('processing');
                orbIcon.textContent = '🎤';
                status.textContent = 'Error - tap to try again';
            }
        }
        
        function addMessage(sender, text, type) {
            const div = document.createElement('div');
            div.className = 'message ' + type;
            div.innerHTML = '<div class="message-label">' + sender + '</div>' + text;
            transcript.appendChild(div);
            transcript.scrollTop = transcript.scrollHeight;
        }
    </script>
</body>
</html>
"""


@app.post("/voice/query")
async def voice_query(audio: UploadFile = File(...)):
    """Process voice query and return text response."""
    if voice_bot is None:
        raise HTTPException(status_code=503, detail="Voice bot not initialized")
    
    import openai
    
    client = openai.AsyncOpenAI()
    
    # Read audio file
    audio_data = await audio.read()
    
    # Convert webm to wav if needed (OpenAI Whisper supports webm directly)
    audio_file = io.BytesIO(audio_data)
    audio_file.name = "recording.webm"
    
    # Transcribe
    try:
        transcript = await client.audio.transcriptions.create(
            model="whisper-1",
            file=audio_file
        )
        user_text = transcript.text
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Transcription failed: {e}")
    
    # Get vision context
    image_b64 = await vision_provider.get_frame_base64() if vision_provider else None
    detection_summary = await vision_provider.get_detection_summary() if vision_provider else "No camera available"
    
    # Build messages
    system_prompt = """You are K1, a friendly robot assistant with vision capabilities.
You can see through your camera and describe what you observe.

When asked about what you see:
1. Describe the scene naturally and conversationally
2. Mention specific objects with their positions
3. Be helpful but keep responses concise (2-3 sentences)

Keep responses short and conversational - you're speaking out loud!"""

    if image_b64:
        messages = [
            {"role": "system", "content": system_prompt},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": f"[{detection_summary}]\n\nUser: {user_text}"},
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{image_b64}",
                            "detail": "high"
                        }
                    }
                ]
            }
        ]
    else:
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_text}
        ]
    
    # Get response
    try:
        response = await client.chat.completions.create(
            model="gpt-4o",
            messages=messages,
            max_tokens=300
        )
        response_text = response.choices[0].message.content
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"LLM failed: {e}")
    
    return {
        "transcription": user_text,
        "response": response_text,
        "detections": detection_summary
    }


@app.get("/voice/speak")
async def speak(text: str):
    """Convert text to speech and return audio."""
    import openai
    
    client = openai.AsyncOpenAI()
    
    try:
        response = await client.audio.speech.create(
            model="tts-1",
            voice="nova",
            input=text
        )
        
        return StreamingResponse(
            io.BytesIO(response.content),
            media_type="audio/mpeg",
            headers={"Content-Disposition": "inline; filename=speech.mp3"}
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"TTS failed: {e}")


@app.get("/camera/latest")
async def camera_latest():
    """Get the latest camera frame as JPEG."""
    if vision_provider is None or vision_provider.latest_frame is None:
        # Return a placeholder image
        placeholder = np.zeros((480, 640, 3), dtype=np.uint8)
        cv2.putText(placeholder, "No Camera Feed", (180, 240), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        _, buffer = cv2.imencode('.jpg', placeholder)
        return StreamingResponse(
            io.BytesIO(buffer.tobytes()),
            media_type="image/jpeg"
        )
    
    _, buffer = cv2.imencode('.jpg', vision_provider.latest_frame, 
                            [cv2.IMWRITE_JPEG_QUALITY, 80])
    return StreamingResponse(
        io.BytesIO(buffer.tobytes()),
        media_type="image/jpeg"
    )


@app.get("/health")
async def health():
    """Health check endpoint."""
    return {
        "status": "ok",
        "camera": vision_provider is not None and vision_provider.latest_frame is not None,
        "voice": voice_bot is not None
    }


def run_voice_server(host: str = "0.0.0.0", port: int = 8090):
    """Run the voice assistant server."""
    import uvicorn
    uvicorn.run(app, host=host, port=port)


# Standalone mode with local camera
async def run_standalone(camera_id: int = 0, host: str = "0.0.0.0", port: int = 8090):
    """Run voice assistant with local camera (no ROS2)."""
    global vision_provider, voice_bot
    
    print("🤖 K1 Voice Assistant - Standalone Mode")
    print("=" * 50)
    
    # Initialize vision provider
    vision_provider = VisionFrameProvider()
    voice_bot = SimplifiedVoiceBot(vision_provider)
    
    # Try to load YOLO for detection
    detector = None
    try:
        from ultralytics import YOLO
        detector = YOLO("yolov8n.pt")
        print("✓ YOLO model loaded")
    except Exception as e:
        print(f"⚠ YOLO not available: {e}")
    
    # Open camera
    cap = cv2.VideoCapture(camera_id)
    if not cap.isOpened():
        print(f"❌ Failed to open camera {camera_id}")
        return
    print(f"✓ Camera {camera_id} opened")
    
    # Start camera capture in background
    async def capture_loop():
        while True:
            ret, frame = cap.read()
            if ret:
                detections = []
                if detector:
                    results = detector(frame, verbose=False)
                    for result in results:
                        if result.boxes:
                            for box in result.boxes:
                                detections.append({
                                    'class_name': result.names[int(box.cls[0])],
                                    'confidence': float(box.conf[0])
                                })
                
                await vision_provider.update_frame(frame, detections)
            await asyncio.sleep(0.033)  # ~30 FPS
    
    # Start capture task
    capture_task = asyncio.create_task(capture_loop())
    
    # Run server
    print(f"\n🌐 Voice assistant running at http://{host}:{port}")
    print("📷 Open in browser to use voice interface")
    print("Press Ctrl+C to stop\n")
    
    import uvicorn
    config = uvicorn.Config(app, host=host, port=port, log_level="info")
    server = uvicorn.Server(config)
    
    try:
        await server.serve()
    finally:
        capture_task.cancel()
        cap.release()


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="K1 Voice Assistant")
    parser.add_argument("--camera", type=int, default=0, help="Camera device ID")
    parser.add_argument("--host", type=str, default="0.0.0.0", help="Server host")
    parser.add_argument("--port", type=int, default=8090, help="Server port")
    
    args = parser.parse_args()
    
    asyncio.run(run_standalone(args.camera, args.host, args.port))

