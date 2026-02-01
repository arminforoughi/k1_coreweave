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

# Browserbase research integration
try:
    from workers.research.researcher import research_object, BROWSERBASE_API_KEY
    RESEARCH_AVAILABLE = bool(BROWSERBASE_API_KEY)
except ImportError:
    RESEARCH_AVAILABLE = False
    research_object = None

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
    
    def get_latest_object_crop_b64(self) -> tuple[Optional[str], str, float]:
        """Get the latest detected object crop for research.
        
        Returns:
            tuple: (crop_b64, yolo_class, yolo_confidence) or (None, "", 0) if unavailable
        """
        # Try to get from Redis (object crops stored by detection pipeline)
        if self.redis:
            try:
                # Get the most recent object crop
                crop_data = self.redis.get("latest_object_crop")
                if crop_data:
                    data = json.loads(crop_data)
                    return (
                        data.get("thumbnail_b64", ""),
                        data.get("yolo_class", "unknown"),
                        float(data.get("yolo_confidence", 0))
                    )
            except Exception:
                pass
        
        # Fallback: try HTTP API for latest detection with crop
        try:
            with httpx.Client() as client:
                resp = client.get(
                    f"{self.backend_url}/events/objects",
                    params={"count": 1, "include_crops": True},
                    timeout=5.0
                )
                if resp.status_code == 200:
                    events = resp.json().get("events", [])
                    if events:
                        e = events[0]
                        return (
                            e.get("thumbnail_b64", ""),
                            e.get("yolo_class", e.get("label", "unknown")),
                            float(e.get("yolo_confidence", e.get("similarity", 0)))
                        )
        except Exception:
            pass
        
        return (None, "", 0.0)


class K1VoiceChat:
    """
    Voice chat with K1 robot.
    Uses laptop mic/speaker + vision from robot.
    Now with Browserbase research for deep object identification!
    """
    
    # Keywords that trigger web research
    RESEARCH_KEYWORDS = [
        "research", "look up", "lookup", "search", "find out", "google",
        "tell me more", "what is that", "identify", "what's that",
        "more about", "details", "information about", "info on",
        "learn more", "find more", "who made", "manufacturer",
        "price", "how much", "where can i buy", "specs", "specifications"
    ]
    
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
        
        # Track last research result for follow-up questions
        self.last_research_result = None
    
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
- Use conversational language

You also have a research capability. When the user asks to research an object,
you can search the web for detailed product information, specs, and pricing."""
    
    def _is_research_query(self, text: str) -> bool:
        """Check if the user is asking for web research."""
        text_lower = text.lower()
        return any(kw in text_lower for kw in self.RESEARCH_KEYWORDS)
    
    def do_research(self, speak_updates: bool = True) -> Optional[dict]:
        """Perform Browserbase research on the latest detected object.
        
        Args:
            speak_updates: If True, speak status updates during research
            
        Returns:
            Research result dict or None if research failed/unavailable
        """
        if not RESEARCH_AVAILABLE:
            return None
        
        # Get the latest object crop
        crop_b64, yolo_class, yolo_conf = self.vision.get_latest_object_crop_b64()
        
        if not crop_b64:
            # Try using the current frame if no crop available
            frame_b64 = self.vision.get_latest_frame_b64()
            detections = self.vision.get_latest_detections()
            
            if frame_b64 and detections:
                # Use the highest confidence detection as hint
                best_det = max(detections, key=lambda d: d.get("confidence", 0))
                crop_b64 = frame_b64  # Use full frame
                yolo_class = best_det.get("class_name", "object")
                yolo_conf = best_det.get("confidence", 0)
            else:
                return None
        
        if speak_updates:
            self.speak(f"Researching the {yolo_class}. This may take a few seconds.")
        
        print(f"{Colors.CYAN}🔍 Starting web research for: {yolo_class}{Colors.END}")
        
        try:
            result = research_object(
                thumbnail_b64=crop_b64,
                yolo_hint=yolo_class,
                yolo_confidence=yolo_conf
            )
            
            if result:
                self.last_research_result = result
                print(f"{Colors.GREEN}✓ Research complete: {result.get('label')}{Colors.END}")
                return result
            else:
                print(f"{Colors.YELLOW}⚠ Research returned no results{Colors.END}")
                return None
                
        except Exception as e:
            print(f"{Colors.RED}✗ Research failed: {e}{Colors.END}")
            return None
    
    def format_research_for_speech(self, result: dict) -> str:
        """Format research results into natural speech."""
        parts = []
        
        label = result.get("label", "unknown object")
        confidence = result.get("confidence", 0)
        
        # Main identification
        if confidence >= 0.8:
            parts.append(f"I'm pretty confident this is a {label}.")
        elif confidence >= 0.6:
            parts.append(f"This looks like a {label}.")
        else:
            parts.append(f"I think this might be a {label}, but I'm not entirely sure.")
        
        # Description
        desc = result.get("web_description") or result.get("description")
        if desc and len(desc) < 200:
            parts.append(desc)
        
        # Manufacturer
        manufacturer = result.get("manufacturer")
        if manufacturer:
            parts.append(f"It's made by {manufacturer}.")
        
        # Price
        price = result.get("price")
        if price:
            parts.append(f"The price is around {price}.")
        
        # Key specs (just a couple)
        specs = result.get("specs", [])
        if specs and len(specs) >= 2:
            parts.append(f"Key specs include {specs[0]} and {specs[1]}.")
        elif specs:
            parts.append(f"One key spec is {specs[0]}.")
        
        # Safety info
        safety = result.get("safety_info")
        if safety and "high" in safety.lower():
            parts.append(f"Safety note: {safety}")
        
        return " ".join(parts)
    
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
        """Get AI response, optionally with vision context.
        
        If this is a research query and Browserbase is available, performs
        web research first and includes results in the response.
        """
        
        # Check if this is a research request
        if self._is_research_query(user_text) and RESEARCH_AVAILABLE:
            result = self.do_research(speak_updates=True)
            if result:
                # Return formatted research results
                response_text = self.format_research_for_speech(result)
                self.conversation_history.append({"role": "user", "content": user_text})
                self.conversation_history.append({"role": "assistant", "content": response_text})
                return response_text
            else:
                # Fall through to normal response if research failed
                pass
        
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
        print(f"\n{Colors.BOLD}Example commands:{Colors.END}")
        print(f"  • {Colors.YELLOW}\"What do you see?\"{Colors.END} - Describe current view")
        if RESEARCH_AVAILABLE:
            print(f"  • {Colors.YELLOW}\"Research that object\"{Colors.END} - Deep web search")
            print(f"  • {Colors.YELLOW}\"Tell me more about that\"{Colors.END} - Get product details")
            print(f"  • {Colors.YELLOW}\"What is that thing?\"{Colors.END} - Identify + research")
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
        
        # Check research capability
        if RESEARCH_AVAILABLE:
            print(f"{Colors.GREEN}✓ Web research enabled (Browserbase){Colors.END}")
        else:
            print(f"{Colors.YELLOW}⚠ Web research disabled (set BROWSERBASE_API_KEY to enable){Colors.END}")
        
        print()
        
        # Greeting
        if RESEARCH_AVAILABLE:
            greeting = "Hi! I'm K1. Ask me what I can see, or say 'research that' to learn more about an object!"
        else:
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

