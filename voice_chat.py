#!/usr/bin/env python3
"""
K1 Voice Chat - Laptop-based voice assistant
Uses local microphone/speaker + vision data from robot

Run this on your laptop while the Jetson sends camera data to the backend.
"""

import os
import io
import sys
import time
import json
import base64
import asyncio
import threading
import tempfile
import wave
import struct
from typing import Optional
from datetime import datetime
from dotenv import load_dotenv

import numpy as np
import httpx
import redis

load_dotenv()

# Audio recording/playback
try:
    import sounddevice as sd
    AUDIO_BACKEND = "sounddevice"
except ImportError:
    try:
        import pyaudio
        AUDIO_BACKEND = "pyaudio"
    except ImportError:
        print("❌ No audio backend found. Install: pip install sounddevice")
        sys.exit(1)

# OpenAI
from openai import OpenAI

# Colors for terminal output
class Colors:
    CYAN = '\033[96m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    MAGENTA = '\033[95m'
    BOLD = '\033[1m'
    END = '\033[0m'


class AudioRecorder:
    """Records audio from microphone."""
    
    def __init__(self, sample_rate: int = 16000, channels: int = 1):
        self.sample_rate = sample_rate
        self.channels = channels
        self.recording = False
        self.audio_data = []
        
    def start_recording(self):
        """Start recording audio."""
        self.audio_data = []
        self.recording = True
        
        if AUDIO_BACKEND == "sounddevice":
            self._record_sounddevice()
        else:
            self._record_pyaudio()
    
    def _record_sounddevice(self):
        """Record using sounddevice (blocking until stop)."""
        def callback(indata, frames, time_info, status):
            if self.recording:
                self.audio_data.append(indata.copy())
        
        self.stream = sd.InputStream(
            samplerate=self.sample_rate,
            channels=self.channels,
            dtype='int16',
            callback=callback
        )
        self.stream.start()
    
    def _record_pyaudio(self):
        """Record using pyaudio."""
        import pyaudio
        self.pa = pyaudio.PyAudio()
        self.stream = self.pa.open(
            format=pyaudio.paInt16,
            channels=self.channels,
            rate=self.sample_rate,
            input=True,
            frames_per_buffer=1024
        )
        
        def record_thread():
            while self.recording:
                data = self.stream.read(1024, exception_on_overflow=False)
                self.audio_data.append(np.frombuffer(data, dtype=np.int16))
        
        self.record_thread = threading.Thread(target=record_thread)
        self.record_thread.start()
    
    def stop_recording(self) -> bytes:
        """Stop recording and return WAV bytes."""
        self.recording = False
        
        if AUDIO_BACKEND == "sounddevice":
            self.stream.stop()
            self.stream.close()
        else:
            self.record_thread.join()
            self.stream.stop_stream()
            self.stream.close()
            self.pa.terminate()
        
        if not self.audio_data:
            return b""
        
        # Combine audio chunks
        audio = np.concatenate(self.audio_data)
        
        # Convert to WAV bytes
        wav_buffer = io.BytesIO()
        with wave.open(wav_buffer, 'wb') as wav_file:
            wav_file.setnchannels(self.channels)
            wav_file.setsampwidth(2)  # 16-bit
            wav_file.setframerate(self.sample_rate)
            wav_file.writeframes(audio.tobytes())
        
        return wav_buffer.getvalue()


class AudioPlayer:
    """Plays audio through speakers."""
    
    def play_mp3(self, mp3_data: bytes):
        """Play MP3 audio data."""
        # Save to temp file and play
        with tempfile.NamedTemporaryFile(suffix='.mp3', delete=False) as f:
            f.write(mp3_data)
            temp_path = f.name
        
        try:
            if sys.platform == "darwin":
                os.system(f"afplay {temp_path}")
            elif sys.platform == "linux":
                os.system(f"mpg123 -q {temp_path} 2>/dev/null || ffplay -nodisp -autoexit {temp_path} 2>/dev/null")
            else:
                os.system(f"ffplay -nodisp -autoexit {temp_path} 2>/dev/null")
        finally:
            os.unlink(temp_path)


class VisionClient:
    """Gets vision data from the robot via backend API or Redis."""
    
    def __init__(self, backend_url: str = "http://localhost:8000", redis_url: str = "redis://localhost:6379"):
        self.backend_url = backend_url
        self.redis_url = redis_url
        self._redis = None
        
    @property
    def redis(self):
        if self._redis is None:
            try:
                self._redis = redis.from_url(self.redis_url, decode_responses=True)
                self._redis.ping()
            except:
                self._redis = None
        return self._redis
    
    def get_latest_frame_b64(self) -> Optional[str]:
        """Get the latest camera frame as base64."""
        # Try Redis first (faster)
        if self.redis:
            try:
                frame_b64 = self.redis.get("latest_frame")
                if frame_b64:
                    return frame_b64
            except:
                pass
        
        # Fall back to HTTP API
        try:
            with httpx.Client() as client:
                resp = client.get(f"{self.backend_url}/stream/snapshot", timeout=5.0)
                if resp.status_code == 200:
                    return base64.b64encode(resp.content).decode()
        except:
            pass
        
        return None
    
    def get_latest_detections(self) -> list:
        """Get the latest object detections."""
        # Try Redis first
        if self.redis:
            try:
                det_json = self.redis.get("latest_detections")
                if det_json:
                    return json.loads(det_json)
            except:
                pass
        
        # Fall back to HTTP API
        try:
            with httpx.Client() as client:
                resp = client.get(f"{self.backend_url}/events/objects", params={"count": 20}, timeout=5.0)
                if resp.status_code == 200:
                    events = resp.json().get("events", [])
                    # Get unique tracks
                    seen = set()
                    detections = []
                    for e in events:
                        track_id = e.get("track_id")
                        if track_id and track_id not in seen:
                            seen.add(track_id)
                            detections.append({
                                "class_name": e.get("label", e.get("yolo_class", "unknown")),
                                "confidence": float(e.get("similarity", e.get("yolo_confidence", 0))),
                                "state": e.get("state", "unknown"),
                                "bbox": e.get("bbox", []),
                            })
                    return detections
        except Exception as e:
            pass
        
        return []
    
    def format_detections(self, detections: list) -> str:
        """Format detections as human-readable text."""
        if not detections:
            return "No objects currently detected in view."
        
        # Group by class
        counts = {}
        for det in detections:
            cls = det.get("class_name", "unknown")
            if cls not in counts:
                counts[cls] = {"count": 0, "max_conf": 0, "states": []}
            counts[cls]["count"] += 1
            counts[cls]["max_conf"] = max(counts[cls]["max_conf"], det.get("confidence", 0))
            counts[cls]["states"].append(det.get("state", "unknown"))
        
        parts = []
        for cls, info in sorted(counts.items(), key=lambda x: -x[1]["max_conf"]):
            count = info["count"]
            conf = int(info["max_conf"] * 100)
            plural = "s" if count > 1 else ""
            parts.append(f"{count} {cls}{plural} ({conf}% confidence)")
        
        return "Currently detecting: " + ", ".join(parts)


class K1VoiceChat:
    """
    Voice chat with K1 robot.
    Uses laptop mic/speaker + vision from robot.
    """
    
    def __init__(
        self,
        backend_url: str = "http://localhost:8000",
        redis_url: str = "redis://localhost:6379",
    ):
        # Validate OpenAI key
        if not os.getenv("OPENAI_API_KEY"):
            raise ValueError("OPENAI_API_KEY environment variable is required")
        
        self.openai = OpenAI()
        self.recorder = AudioRecorder()
        self.player = AudioPlayer()
        self.vision = VisionClient(backend_url, redis_url)
        
        self.conversation_history = []
        self.system_prompt = self._get_system_prompt()
    
    def _get_system_prompt(self) -> str:
        return """You are K1, a friendly robot assistant with vision capabilities.
You can see through your camera and describe what you observe when the user asks.

When asked about what you see:
1. Describe the scene naturally and conversationally  
2. Mention specific objects with their approximate positions
3. If you see people, describe what they appear to be doing
4. Be helpful and informative but keep responses concise (2-4 sentences)

For general conversation:
- Be friendly and personable
- Keep responses concise since you're speaking out loud
- You can discuss anything the user wants to talk about

Important: Your responses will be spoken aloud, so:
- Keep them short and natural
- Avoid lists or bullet points
- Use conversational language"""
    
    def transcribe(self, audio_bytes: bytes) -> str:
        """Transcribe audio to text using Whisper."""
        audio_file = io.BytesIO(audio_bytes)
        audio_file.name = "recording.wav"
        
        transcript = self.openai.audio.transcriptions.create(
            model="whisper-1",
            file=audio_file
        )
        return transcript.text
    
    def get_response(self, user_text: str, include_vision: bool = True) -> str:
        """Get AI response, optionally with vision context."""
        
        # Check if this is a vision-related query
        vision_keywords = ["see", "look", "looking", "visible", "camera", "what", 
                         "describe", "show", "front", "around", "detect", "there"]
        is_vision_query = any(kw in user_text.lower() for kw in vision_keywords)
        
        # Build message content
        if is_vision_query and include_vision:
            # Get vision data
            frame_b64 = self.vision.get_latest_frame_b64()
            detections = self.vision.get_latest_detections()
            detection_text = self.vision.format_detections(detections)
            
            if frame_b64:
                # Include image in the message
                content = [
                    {"type": "text", "text": f"[Detection summary: {detection_text}]\n\nUser: {user_text}"},
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{frame_b64}",
                            "detail": "high"
                        }
                    }
                ]
            else:
                content = f"[No camera feed available. Detection summary: {detection_text}]\n\nUser: {user_text}"
        else:
            content = user_text
        
        # Build messages
        messages = [{"role": "system", "content": self.system_prompt}]
        messages.extend(self.conversation_history[-10:])  # Last 10 messages for context
        messages.append({"role": "user", "content": content})
        
        # Get response
        response = self.openai.chat.completions.create(
            model="gpt-4o",
            messages=messages,
            max_tokens=300,
            temperature=0.7,
        )
        
        assistant_text = response.choices[0].message.content
        
        # Update conversation history (store text only, not images)
        self.conversation_history.append({"role": "user", "content": user_text})
        self.conversation_history.append({"role": "assistant", "content": assistant_text})
        
        return assistant_text
    
    def speak(self, text: str):
        """Convert text to speech and play it."""
        response = self.openai.audio.speech.create(
            model="tts-1",
            voice="nova",
            input=text
        )
        
        self.player.play_mp3(response.content)
    
    def run_interactive(self):
        """Run interactive voice chat loop."""
        print(f"\n{Colors.BOLD}{Colors.CYAN}🤖 K1 Voice Chat{Colors.END}")
        print(f"{Colors.CYAN}{'=' * 50}{Colors.END}")
        print(f"Press {Colors.BOLD}Enter{Colors.END} to start speaking, {Colors.BOLD}Enter{Colors.END} again to stop.")
        print(f"Say {Colors.BOLD}'quit'{Colors.END} or {Colors.BOLD}'exit'{Colors.END} to end the session.")
        print(f"Ask things like: {Colors.YELLOW}\"What do you see?\"{Colors.END}")
        print(f"{Colors.CYAN}{'=' * 50}{Colors.END}\n")
        
        # Check vision connection
        detections = self.vision.get_latest_detections()
        frame = self.vision.get_latest_frame_b64()
        
        if frame:
            print(f"{Colors.GREEN}✓ Camera feed connected{Colors.END}")
            if detections:
                print(f"{Colors.GREEN}✓ {len(detections)} objects detected{Colors.END}")
        else:
            print(f"{Colors.YELLOW}⚠ No camera feed available. Make sure the robot/backend is running.{Colors.END}")
        
        print()
        
        # Greeting
        greeting = "Hi! I'm K1. Ask me what I can see, or let's just chat!"
        print(f"{Colors.GREEN}K1: {greeting}{Colors.END}")
        self.speak(greeting)
        
        while True:
            try:
                # Wait for user to press Enter
                input(f"\n{Colors.MAGENTA}[Press Enter to speak...]{Colors.END}")
                
                # Start recording
                print(f"{Colors.RED}🎤 Recording... (press Enter to stop){Colors.END}")
                self.recorder.start_recording()
                
                # Wait for Enter to stop
                input()
                
                # Stop recording
                audio_bytes = self.recorder.stop_recording()
                
                if len(audio_bytes) < 1000:
                    print(f"{Colors.YELLOW}(No audio detected, try again){Colors.END}")
                    continue
                
                # Transcribe
                print(f"{Colors.CYAN}Processing...{Colors.END}")
                user_text = self.transcribe(audio_bytes)
                
                if not user_text.strip():
                    print(f"{Colors.YELLOW}(Couldn't understand, try again){Colors.END}")
                    continue
                
                print(f"{Colors.BOLD}You: {user_text}{Colors.END}")
                
                # Check for exit commands
                if user_text.lower().strip() in ["quit", "exit", "bye", "goodbye"]:
                    farewell = "Goodbye! It was nice chatting with you."
                    print(f"{Colors.GREEN}K1: {farewell}{Colors.END}")
                    self.speak(farewell)
                    break
                
                # Get AI response
                response_text = self.get_response(user_text)
                print(f"{Colors.GREEN}K1: {response_text}{Colors.END}")
                
                # Speak response
                self.speak(response_text)
                
            except KeyboardInterrupt:
                print(f"\n{Colors.CYAN}Session ended.{Colors.END}")
                break
            except Exception as e:
                print(f"{Colors.RED}Error: {e}{Colors.END}")
                continue


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="K1 Voice Chat - Talk to your robot")
    parser.add_argument("--backend", type=str, default="http://localhost:8000",
                       help="Backend API URL (default: http://localhost:8000)")
    parser.add_argument("--redis", type=str, default="redis://localhost:6379",
                       help="Redis URL (default: redis://localhost:6379)")
    
    args = parser.parse_args()
    
    try:
        chat = K1VoiceChat(
            backend_url=args.backend,
            redis_url=args.redis,
        )
        chat.run_interactive()
    except ValueError as e:
        print(f"{Colors.RED}Error: {e}{Colors.END}")
        print(f"\nSet your OpenAI API key:")
        print(f"  export OPENAI_API_KEY=sk-...")
        sys.exit(1)


if __name__ == "__main__":
    main()

