from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse
import uvicorn
import json
import asyncio
import numpy as np
import wave
import tempfile
import os
import base64
from collections import deque
from azure.cognitiveservices.speech import SpeechConfig, AudioConfig, SpeechRecognizer, ResultReason
from openai import OpenAI
from dotenv import load_dotenv
import logging
import time
from typing import Optional, Dict, Any
import threading
from concurrent.futures import ThreadPoolExecutor

# Load environment variables
load_dotenv()

# Configuration
AZURE_SPEECH_KEY = os.getenv("AZURE_SPEECH_KEY")
AZURE_REGION = os.getenv("AZURE_REGION")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

USE_REAL_APIS = True

if USE_REAL_APIS:
    openai_client = OpenAI(api_key=OPENAI_API_KEY)
    print("✅ Using real APIs")
else:
    print("🧪 Using fake APIs for testing")

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI()

# Global state
active_sessions: Dict[str, 'SimpleVoiceSession'] = {}

# Thread pool for CPU-intensive tasks
executor = ThreadPoolExecutor(max_workers=4)

class SimpleVoiceSession:
    def __init__(self, websocket, session_id):
        self.websocket = websocket
        self.session_id = session_id
        self.audio_buffer = deque(maxlen=1000)  # Limit buffer size
        self.is_playing_ai = False
        self.is_processing = False
        self.temp_files = []
        
        # Audio parameters - optimized for Vietnamese
        self.SAMPLE_RATE = 16000
        self.CHUNK_DURATION = 2.0  # Reduced for faster response
        self.MIN_AUDIO_LENGTH = 0.5  # Reduced minimum length
        self.VOLUME_THRESHOLD = 300  # Lowered threshold
        self.MAX_SILENCE_DURATION = 1.5  # Auto-stop after silence
        
        # Speech detection
        self.speech_buffer = []
        self.last_speech_time = None
        self.is_recording = False
        self.silence_start = None
        
        # Performance optimization
        self.processing_lock = asyncio.Lock()
        self._is_active = True
        
        logger.info(f"SimpleVoiceSession created: {session_id}")

    async def add_audio_data(self, audio_data: str):
        """Add audio data to buffer with improved error handling"""
        if not self.is_recording or self.is_processing:
            return
            
        try:
            # Decode base64 audio data
            audio_bytes = base64.b64decode(audio_data)
            
            # Check if we have WebM/Opus data (need to convert)
            if audio_bytes.startswith(b'webm') or audio_bytes.startswith(b'\x1a\x45\xdf\xa3'):
                logger.warning("Received WebM data, skipping (need PCM conversion)")
                return
            
            # Handle different audio formats
            try:
                # Try as 16-bit PCM first
                if len(audio_bytes) % 2 == 0:
                    audio_array = np.frombuffer(audio_bytes, dtype=np.int16)
                else:
                    # Odd number of bytes, pad with zero
                    audio_bytes += b'\x00'
                    audio_array = np.frombuffer(audio_bytes, dtype=np.int16)
            except ValueError as e:
                logger.error(f"Failed to parse audio data: {e}")
                return
            
            if len(audio_array) == 0:
                return
                
            # Add to speech buffer
            self.speech_buffer.extend(audio_array)
            
            # Check volume level for silence detection
            rms = np.sqrt(np.mean(np.square(audio_array.astype(np.float32))))
            
            if rms > self.VOLUME_THRESHOLD:
                self.last_speech_time = time.time()
                if self.silence_start:
                    self.silence_start = None
            else:
                if not self.silence_start and self.last_speech_time:
                    self.silence_start = time.time()
            
            # Check if we should process
            duration = len(self.speech_buffer) / self.SAMPLE_RATE
            
            # Auto-process on silence or max duration
            should_process = (
                duration >= self.CHUNK_DURATION or
                (self.silence_start and 
                 time.time() - self.silence_start >= self.MAX_SILENCE_DURATION and
                 duration >= self.MIN_AUDIO_LENGTH)
            )
            
            if should_process:
                # Process in background to avoid blocking
                asyncio.create_task(self.process_speech_buffer())
                
        except Exception as e:
            logger.error(f"Error adding audio data: {e}")

    async def process_speech_buffer(self):
        """Process accumulated speech buffer with improved performance"""
        async with self.processing_lock:
            if self.is_processing or not self.speech_buffer:
                return
                
            self.is_processing = True
            wav_path = None
            
            try:
                # Get audio data
                audio_data = np.array(self.speech_buffer, dtype=np.int16)
                duration = len(audio_data) / self.SAMPLE_RATE
                
                logger.info(f"Processing {duration:.2f}s of audio")
                
                # Clear buffer for next recording
                self.speech_buffer.clear()
                self.silence_start = None
                
                # Skip if too short
                if duration < self.MIN_AUDIO_LENGTH:
                    logger.info("Audio too short, skipping")
                    return
                
                # Check if audio has meaningful content
                rms = np.sqrt(np.mean(np.square(audio_data.astype(np.float32))))
                if rms < self.VOLUME_THRESHOLD:
                    logger.info(f"Audio too quiet (RMS: {rms:.1f}), skipping")
                    return
                
                # Save to WAV file in temp directory
                wav_path = tempfile.NamedTemporaryFile(suffix=".wav", delete=False).name
                self.temp_files.append(wav_path)
                
                # Use loop.run_in_executor for I/O operations
                loop = asyncio.get_event_loop()
                await loop.run_in_executor(
                    executor, 
                    self._save_wav_file, 
                    wav_path, 
                    audio_data
                )
                
                logger.info(f"Audio saved to: {wav_path} ({os.path.getsize(wav_path)} bytes)")
                
                # Speech to Text (in executor)
                transcript = await self.speech_to_text(wav_path)
                if not transcript or len(transcript.strip()) < 2:
                    logger.info("No meaningful transcript received")
                    return
                
                logger.info(f"Transcript: {transcript}")
                
                # Generate AI response (in executor)
                ai_response = await self.generate_ai_response(transcript)
                logger.info(f"AI Response: {ai_response}")
                
                # Send transcript and response to client
                await self.websocket.send_text(json.dumps({
                    "type": "transcript",
                    "transcript": transcript,
                    "response": ai_response
                }))
                
                # Generate TTS audio and send to client
                await self.text_to_speech_and_send(ai_response)
                
            except Exception as e:
                logger.error(f"Error processing speech: {e}")
                import traceback
                traceback.print_exc()
                
                # Send error to client
                try:
                    await self.websocket.send_text(json.dumps({
                        "type": "error",
                        "message": "Có lỗi xảy ra khi xử lý âm thanh"
                    }))
                except:
                    pass
            finally:
                self.is_processing = False
                if wav_path:
                    # Schedule cleanup after a delay
                    asyncio.create_task(self._cleanup_file_delayed(wav_path, 2.0))

    def _save_wav_file(self, wav_path: str, audio_data: np.ndarray):
        """Save audio data to WAV file (runs in executor)"""
        with wave.open(wav_path, "wb") as wf:
            wf.setnchannels(1)  # Mono
            wf.setsampwidth(2)  # 16-bit
            wf.setframerate(self.SAMPLE_RATE)  # 16kHz
            wf.writeframes(audio_data.tobytes())

    async def speech_to_text(self, wav_path: str) -> Optional[str]:
        """Azure Speech to Text with improved error handling"""
        try:
            if USE_REAL_APIS:
                loop = asyncio.get_event_loop()
                return await loop.run_in_executor(executor, self._azure_stt, wav_path)
            else:
                # Fake STT for testing
                await asyncio.sleep(0.3)
                return "Test transcript từ fake STT"
        except Exception as e:
            logger.error(f"Speech to text error: {e}")
            return None

    def _azure_stt(self, wav_path: str) -> Optional[str]:
        """Azure Speech Recognition (runs in executor)"""
        try:
            speech_config = SpeechConfig(subscription=AZURE_SPEECH_KEY, region=AZURE_REGION)
            speech_config.speech_recognition_language = "vi-VN"
            
            # Optimized settings for Vietnamese
            speech_config.set_property_by_name("Speech_SegmentationSilenceTimeoutMs", "1500")
            speech_config.set_property_by_name("Speech_SegmentationMaxSilenceTimeoutMs", "3000")
            speech_config.set_property_by_name("Speech_EndSilenceTimeoutMs", "1000")
            
            audio_config = AudioConfig(filename=wav_path)
            recognizer = SpeechRecognizer(speech_config=speech_config, audio_config=audio_config)
            
            logger.info("Starting Azure STT...")
            result = recognizer.recognize_once()
            
            if result.reason == ResultReason.RecognizedSpeech:
                logger.info(f"✅ Azure transcript: '{result.text}'")
                return result.text
            elif result.reason == ResultReason.NoMatch:
                logger.warning("❌ No speech detected")
                return None
            elif result.reason == ResultReason.Canceled:
                logger.error(f"❌ Azure STT error: {result.cancellation_details}")
                return None
            
        except Exception as e:
            logger.error(f"Azure STT exception: {e}")
            return None

    async def generate_ai_response(self, transcript: str) -> str:
        """Generate AI response with improved error handling"""
        try:
            if USE_REAL_APIS:
                loop = asyncio.get_event_loop()
                return await loop.run_in_executor(executor, self._openai_chat, transcript)
            else:
                await asyncio.sleep(0.3)
                return f"Phản hồi test cho: '{transcript}'"
        except Exception as e:
            logger.error(f"AI response error: {e}")
            return "Xin lỗi, có lỗi xảy ra khi tạo phản hồi."

    def _openai_chat(self, transcript: str) -> str:
        """OpenAI Chat (runs in executor)"""
        try:
            response = openai_client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {
                        "role": "system", 
                        "content": "Bạn là trợ lý AI thông minh, trả lời ngắn gọn và hữu ích bằng tiếng Việt. Giữ câu trả lời trong khoảng 1-2 câu."
                    },
                    {"role": "user", "content": transcript}
                ],
                max_tokens=100,  # Reduced for faster response
                temperature=0.7,
                timeout=10  # Add timeout
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            logger.error(f"OpenAI error: {e}")
            return "Có lỗi xảy ra với AI."

    async def text_to_speech_and_send(self, text: str):
        """Convert text to speech and send to client with improved performance"""
        try:
            logger.info(f"🗣️ Starting TTS for: {text[:50]}...")
            
            if USE_REAL_APIS:
                loop = asyncio.get_event_loop()
                audio_data = await loop.run_in_executor(executor, self._openai_tts, text)
            else:
                # Fake TTS
                audio_data = await loop.run_in_executor(executor, self._create_fake_audio)
            
            if not audio_data:
                logger.error("No audio data generated")
                return
            
            # Convert audio to base64 for transmission
            audio_base64 = base64.b64encode(audio_data).decode('utf-8')
            
            # Send audio data to client
            await self.websocket.send_text(json.dumps({
                "type": "audio",
                "audio_data": audio_base64,
                "format": "wav"
            }))
            
            logger.info("✅ TTS audio sent to client")
            
        except Exception as e:
            logger.error(f"TTS error: {e}")

    def _openai_tts(self, text: str) -> Optional[bytes]:
        """OpenAI Text to Speech (runs in executor)"""
        try:
            response = openai_client.audio.speech.create(
                model="tts-1",
                voice="nova",
                input=text,
                response_format="wav",
                timeout=15  # Add timeout
            )
            logger.info("✅ OpenAI TTS completed")
            return response.content
        except Exception as e:
            logger.error(f"OpenAI TTS error: {e}")
            return None

    def _create_fake_audio(self) -> Optional[bytes]:
        """Create fake audio for testing (runs in executor)"""
        try:
            duration = 2.0
            sample_rate = 16000
            samples = int(duration * sample_rate)
            
            # Generate simple tone
            t = np.linspace(0, duration, samples)
            audio = (np.sin(2 * np.pi * 440 * t) * 0.1).astype(np.float32)
            audio_int16 = (audio * 32767).astype(np.int16)
            
            # Create WAV file in memory
            import io
            wav_buffer = io.BytesIO()
            
            with wave.open(wav_buffer, 'wb') as wf:
                wf.setnchannels(1)
                wf.setsampwidth(2)
                wf.setframerate(sample_rate)
                wf.writeframes(audio_int16.tobytes())
            
            wav_buffer.seek(0)
            return wav_buffer.read()
            
        except Exception as e:
            logger.error(f"Fake audio error: {e}")
            return None

    async def _cleanup_file_delayed(self, file_path: str, delay: float):
        """Clean up temp file after delay"""
        try:
            await asyncio.sleep(delay)
            if file_path in self.temp_files:
                self.temp_files.remove(file_path)
            if os.path.exists(file_path):
                os.unlink(file_path)
                logger.debug(f"🗑️ Cleaned up: {file_path}")
        except Exception as e:
            logger.error(f"Cleanup error: {e}")

    async def cleanup(self):
        """Clean up session with improved cleanup"""
        try:
            logger.info(f"Cleaning up session {self.session_id}")
            self._is_active = False
            
            # Stop any ongoing processing
            self.is_processing = False
            self.is_recording = False
            
            # Clean up temp files
            cleanup_tasks = []
            for file_path in self.temp_files[:]:
                if os.path.exists(file_path):
                    cleanup_tasks.append(self._cleanup_file_delayed(file_path, 0))
            
            if cleanup_tasks:
                await asyncio.gather(*cleanup_tasks, return_exceptions=True)
            
        except Exception as e:
            logger.error(f"Session cleanup error: {e}")

    def is_active(self) -> bool:
        return self._is_active

@app.get("/")
async def root():
    return {
        "message": "Simple Voice Chat API",
        "websocket": "ws://localhost:8000/ws",
        "active_sessions": len(active_sessions),
        "status": "running"
    }

@app.get("/health")
async def health():
    return {
        "status": "healthy",
        "active_sessions": len(active_sessions),
        "use_real_apis": USE_REAL_APIS,
        "azure_configured": bool(AZURE_SPEECH_KEY and AZURE_REGION),
        "openai_configured": bool(OPENAI_API_KEY)
    }

@app.get("/sessions")
async def list_sessions():
    return {
        "active_sessions": list(active_sessions.keys()),
        "count": len(active_sessions)
    }

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    session_id = f"session_{len(active_sessions)}_{int(time.time())}"
    session = None
    
    try:
        await websocket.accept()
        logger.info(f"✅ WebSocket connected: {session_id}")
        
        # Create session
        session = SimpleVoiceSession(websocket, session_id)
        active_sessions[session_id] = session
        
        # Send connection confirmation
        await websocket.send_text(json.dumps({
            "type": "connected",
            "session_id": session_id,
            "server_info": {
                "use_real_apis": USE_REAL_APIS,
                "sample_rate": session.SAMPLE_RATE,
                "chunk_duration": session.CHUNK_DURATION
            }
        }))
        
        # Handle messages
        while session.is_active():
            try:
                # Set timeout for receiving messages
                message = await asyncio.wait_for(
                    websocket.receive_text(), 
                    timeout=30.0
                )
                data = json.loads(message)
                
                if data["type"] == "audio_data":
                    # Receive audio data from client
                    logger.debug("📨 Received audio data from client")
                    await session.add_audio_data(data["audio_data"])
                    
                elif data["type"] == "start_recording":
                    logger.info("🎤 Client started recording")
                    session.is_recording = True
                    session.speech_buffer.clear()
                    session.last_speech_time = None
                    session.silence_start = None
                    
                elif data["type"] == "stop_recording":
                    logger.info("🛑 Client stopped recording")
                    session.is_recording = False
                    if session.speech_buffer and len(session.speech_buffer) > 0:
                        # Process remaining audio
                        await session.process_speech_buffer()
                        
                elif data["type"] == "ping":
                    await websocket.send_text(json.dumps({
                        "type": "pong",
                        "timestamp": time.time()
                    }))
                    
                elif data["type"] == "get_status":
                    await websocket.send_text(json.dumps({
                        "type": "status",
                        "session_id": session_id,
                        "is_recording": session.is_recording,
                        "is_processing": session.is_processing,
                        "buffer_length": len(session.speech_buffer),
                        "active": session.is_active()
                    }))
                    
                else:
                    logger.warning(f"Unknown message type: {data['type']}")
                    
            except asyncio.TimeoutError:
                # Send keepalive ping
                try:
                    await websocket.send_text(json.dumps({
                        "type": "keepalive",
                        "timestamp": time.time()
                    }))
                except:
                    break
            except json.JSONDecodeError as e:
                logger.error(f"JSON decode error: {e}")
                await websocket.send_text(json.dumps({
                    "type": "error",
                    "message": "Invalid JSON format"
                }))
            except Exception as e:
                logger.error(f"Message handling error: {e}")
                break
                
    except WebSocketDisconnect:
        logger.info(f"🔌 WebSocket disconnected: {session_id}")
    except Exception as e:
        logger.error(f"WebSocket error: {e}")
    finally:
        # Cleanup
        if session:
            if session_id in active_sessions:
                del active_sessions[session_id]
            await session.cleanup()
        logger.info(f"Session {session_id} ended")

# Graceful shutdown
@app.on_event("shutdown")
async def shutdown_event():
    logger.info("🛑 Server shutting down...")
    
    # Cleanup all active sessions
    cleanup_tasks = []
    for session in active_sessions.values():
        cleanup_tasks.append(session.cleanup())
    
    if cleanup_tasks:
        await asyncio.gather(*cleanup_tasks, return_exceptions=True)
    
    # Shutdown executor
    executor.shutdown(wait=True)
    logger.info("✅ Server shutdown completed")

if __name__ == "__main__":
    print("🚀 Starting Simple Voice Chat API...")
    print("📋 Endpoints:")
    print("   - HTTP: http://localhost:8000")
    print("   - WebSocket: ws://localhost:8000/ws")
    print("   - Health: http://localhost:8000/health")
    print("   - Sessions: http://localhost:8000/sessions")
    print(f"🔧 Configuration:")
    print(f"   - Real APIs: {USE_REAL_APIS}")
    print(f"   - Azure STT: {'✅' if AZURE_SPEECH_KEY else '❌'}")
    print(f"   - OpenAI: {'✅' if OPENAI_API_KEY else '❌'}")
    
    uvicorn.run(
        "sockets:app", 
        host="0.0.0.0", 
        port=8000, 
        reload=True,
        log_level="info"
    )