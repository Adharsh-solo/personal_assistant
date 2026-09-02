"""
Jarvis - Windows Desktop Voice Assistant (Phase 1)
--------------------------------------------------
Runs locally in the background as a system tray application.
Features:
- Continuous wake word detection ("hey_jarvis") via openWakeWord (100% open-source & local)
- Audio speech recording via sounddevice with energy-based VAD
- Faster-Whisper local speech-to-text (GPU accelerated via CUDA with CPU fallback)
- Local LLM query via Ollama (llama3.1:8b)
- Offline Text-to-Speech via pyttsx3 (SAPI5 on Windows)
- Dynamic system tray icon and status updates via pystray
- Silent background logging to jarvis.log
"""

import os
import sys
import time
import logging
import threading
import numpy as np
import sounddevice as sd
import requests
import pyttsx3
import pystray
from PIL import Image, ImageDraw

# ==========================================
# CONFIGURATION
# ==========================================
# openWakeWord Settings (No API key required)
WAKE_WORD_MODEL = "hey_jarvis"   # Pre-trained openWakeWord model ("hey_jarvis", "alexa", etc.)
WAKE_WORD_THRESHOLD = 0.5       # Detection confidence threshold (0.0 to 1.0)

# Whisper STT Settings
WHISPER_MODEL_SIZE = "base"     # "tiny", "base", "small", "medium", "large-v3"
WHISPER_DEVICE = "cuda"         # "cuda" for NVIDIA GPU, "cpu" for fallback
WHISPER_COMPUTE_TYPE = "float16" # "float16" for GPU, "int8" for CPU

# Ollama LLM Settings
OLLAMA_URL = "http://localhost:11434"
OLLAMA_MODEL = "llama3.1:8b"

# Audio Recording Settings
INPUT_DEVICE_INDEX = None       # None uses standard default MME driver (Microsoft Sound Mapper)
SAMPLE_RATE = 16000             # openWakeWord requires 16000 Hz

AUDIO_GAIN = 10.0               # Digital software volume multiplier for low-gain mic hardware
CHUNK_SIZE = 1280               # 1280 samples (~80ms chunk at 16kHz)
SILENCE_THRESHOLD = 0.005       # Energy threshold for silence detection (lowered for gained signal)
SILENCE_DURATION = 1.5          # Seconds of silence to trigger end of speech
MAX_RECORD_SECONDS = 12.0       # Maximum audio recording duration per prompt

# System & Logging
LOG_FILE = "jarvis.log"



# ==========================================
# LOGGING SETUP
# ==========================================
def setup_logging():
    """Configure structured logging to jarvis.log and stdout."""
    log_formatter = logging.Formatter(
        "%(asctime)s [%(levelname)s] %(message)s", datefmt="%Y-%m-%d %H:%M:%S"
    )
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    # Clear existing handlers to prevent duplicates
    logger.handlers.clear()

    # File handler
    file_handler = logging.FileHandler(LOG_FILE, mode="a", encoding="utf-8")
    file_handler.setFormatter(log_formatter)
    logger.addHandler(file_handler)

    # Console handler (active when console is present)
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(log_formatter)
    logger.addHandler(console_handler)

    logging.info("==========================================")
    logging.info("Jarvis Assistant Initializing...")
    logging.info("==========================================")


# ==========================================
# DYNAMIC TRAY ICON GENERATOR
# ==========================================
def create_tray_icon_image(state="idle", width=64, height=64):
    """
    Generates a dynamic 64x64 PIL Image for the tray icon depending on assistant state.
    - idle / listening: Dark Slate background with glowing Cyan core
    - processing / speech / thinking: Dark Slate with glowing Amber core
    - speaking: Dark Slate with glowing Purple core
    - paused / error: Dark Slate with muted Grey or Red core
    """
    image = Image.new("RGBA", (width, height), (0, 0, 0, 0))
    draw = ImageDraw.Draw(image)

    # Outer circle background
    margin = 4
    draw.ellipse(
        [margin, margin, width - margin, height - margin],
        fill=(15, 23, 42, 255),    # Slate 900
        outline=(51, 65, 85, 255),  # Slate 700
        width=2,
    )

    core_margin = 16
    if state == "listening":
        # Cyan Pulse
        draw.ellipse(
            [core_margin - 4, core_margin - 4, width - core_margin + 4, height - core_margin + 4],
            outline=(6, 182, 212, 160),
            width=3,
        )
        draw.ellipse(
            [core_margin, core_margin, width - core_margin, height - core_margin],
            fill=(6, 182, 212, 255),
        )
    elif state in ("recording", "thinking"):
        # Amber Glow
        draw.ellipse(
            [core_margin - 4, core_margin - 4, width - core_margin + 4, height - core_margin + 4],
            outline=(245, 158, 11, 160),
            width=3,
        )
        draw.ellipse(
            [core_margin, core_margin, width - core_margin, height - core_margin],
            fill=(245, 158, 11, 255),
        )
    elif state == "speaking":
        # Purple Wave
        draw.ellipse(
            [core_margin - 4, core_margin - 4, width - core_margin + 4, height - core_margin + 4],
            outline=(168, 85, 247, 160),
            width=3,
        )
        draw.ellipse(
            [core_margin, core_margin, width - core_margin, height - core_margin],
            fill=(168, 85, 247, 255),
        )
    elif state == "paused":
        # Muted Grey
        draw.ellipse(
            [core_margin, core_margin, width - core_margin, height - core_margin],
            fill=(100, 116, 139, 255),
        )
    else:  # error or default
        # Crimson Red
        draw.ellipse(
            [core_margin, core_margin, width - core_margin, height - core_margin],
            fill=(239, 68, 68, 255),
        )

    return image


# ==========================================
# JARVIS ASSISTANT CLASS
# ==========================================
class JarvisAssistant:
    def __init__(self):
        self.state = "idle"
        self.state_text = f"Idle (Listening for '{WAKE_WORD_MODEL}')"
        self.stop_event = threading.Event()
        self.is_paused = False
        self.oww_model = None
        self.whisper_model = None
        self.tray_icon = None

        # GUI Event Callbacks
        self.on_state_change = None
        self.on_user_transcript = None
        self.on_llm_response = None

    def set_state(self, new_state, description):
        """Update assistant state and refresh system tray icon visual and GUI listeners."""
        self.state = new_state
        self.state_text = description
        logging.info(f"State -> {new_state.upper()}: {description}")
        if self.tray_icon:
            self.tray_icon.icon = create_tray_icon_image(new_state)
            self.tray_icon.title = f"Jarvis - {description}"[:120]
        if self.on_state_change:
            try:
                self.on_state_change(new_state, description)
            except Exception as e:
                logging.error(f"Error in on_state_change callback: {e}")

    def init_whisper(self):
        """Lazy load Faster-Whisper model on startup."""
        logging.info(f"Loading Faster-Whisper model ('{WHISPER_MODEL_SIZE}') on device '{WHISPER_DEVICE}'...")
        try:
            from faster_whisper import WhisperModel
            try:
                model = WhisperModel(
                    WHISPER_MODEL_SIZE,
                    device=WHISPER_DEVICE,
                    compute_type=WHISPER_COMPUTE_TYPE,
                )
                logging.info("Faster-Whisper loaded successfully on CUDA GPU.")
                return model
            except Exception as cuda_err:
                logging.warning(f"CUDA initialization failed ({cuda_err}). Falling back to CPU.")
                model = WhisperModel(
                    WHISPER_MODEL_SIZE,
                    device="cpu",
                    compute_type="int8",
                )
                logging.info("Faster-Whisper loaded successfully on CPU.")
                return model
        except Exception as e:
            logging.error(f"Failed to load Faster-Whisper: {e}")
            return None

    def init_openwakeword(self):
        """Initialize openWakeWord engine."""
        logging.info(f"Loading openWakeWord model ('{WAKE_WORD_MODEL}')...")
        try:
            import openwakeword
            from openwakeword.model import Model

            # Ensure pretrained models exist locally
            try:
                openwakeword.utils.download_models()
            except Exception as dl_err:
                logging.warning(f"openWakeWord download check: {dl_err}")

            oww = Model(wakeword_models=[WAKE_WORD_MODEL], inference_framework="onnx")
            logging.info(f"openWakeWord initialized successfully for model '{WAKE_WORD_MODEL}'.")
            return oww
        except Exception as e:
            logging.error(f"Failed to initialize openWakeWord: {e}")
            return None

    def record_audio(self):
        """
        Record audio from microphone following wake word trigger.
        Uses energy-based Voice Activity Detection (VAD) to auto-detect silence.
        """
        logging.info("Listening for user speech input...")
        self.set_state("recording", "Listening to your request...")

        chunk_samples = 1024
        audio_buffer = []
        silence_start = None
        has_spoken = False
        start_time = time.time()

        dev_idx = INPUT_DEVICE_INDEX if INPUT_DEVICE_INDEX is not None else sd.default.device[0]
        try:
            dev_channels = max(1, sd.query_devices(dev_idx, "input")["max_input_channels"])
        except Exception:
            dev_channels = 1

        with sd.InputStream(
            samplerate=SAMPLE_RATE,
            channels=dev_channels,
            dtype="float32",
            device=INPUT_DEVICE_INDEX,
        ) as stream:
            while not self.stop_event.is_set():
                elapsed = time.time() - start_time
                if elapsed >= MAX_RECORD_SECONDS:
                    logging.info("Reached maximum recording limit.")
                    break

                data, overflowed = stream.read(chunk_samples)
                if overflowed:
                    logging.warning("Audio recording buffer overflowed.")

                if dev_channels > 1:
                    audio_chunk = np.mean(data, axis=1).astype(np.float32)
                else:
                    audio_chunk = data.flatten()

                # Apply digital gain multiplier for low-volume mics
                audio_chunk = np.clip(audio_chunk * AUDIO_GAIN, -1.0, 1.0)

                audio_buffer.append(audio_chunk)

                # Compute Root Mean Square (RMS) energy
                rms = np.sqrt(np.mean(audio_chunk**2))

                if rms > SILENCE_THRESHOLD:
                    has_spoken = True
                    silence_start = None
                elif has_spoken:
                    if silence_start is None:
                        silence_start = time.time()
                    elif time.time() - silence_start >= SILENCE_DURATION:
                        logging.info("Speech paused / silence threshold met.")
                        break

        if not audio_buffer:
            return None

        full_audio = np.concatenate(audio_buffer, axis=0)
        return full_audio if has_spoken else None

    def transcribe_audio(self, audio_data):
        """Transcribe float32 audio numpy array using Faster-Whisper."""
        if audio_data is None or len(audio_data) == 0:
            return ""

        self.set_state("thinking", "Transcribing speech...")
        try:
            segments, info = self.whisper_model.transcribe(audio_data, beam_size=5)
            transcript = " ".join([segment.text for segment in segments]).strip()
            logging.info(f"Transcribed Text: '{transcript}'")
            return transcript
        except Exception as e:
            logging.error(f"Error during transcription: {e}")
            return ""

    def get_llm_response(self, text):
        """Send transcribed prompt to Ollama local API."""
        if not text:
            return "I couldn't hear what you said. Please try again."

        self.set_state("thinking", "Generating LLM response...")
        logging.info(f"Sending request to Ollama model '{OLLAMA_MODEL}'...")

        payload = {
            "model": OLLAMA_MODEL,
            "messages": [
                {
                    "role": "system",
                    "content": (
                        "You are Jarvis, an intelligent, helpful, and concise desktop voice assistant. "
                        "Keep your responses brief, conversational, and easy to speak out loud (1 to 3 sentences maximum). "
                        "Do not include code blocks, bullet lists, markdown, or visual formatting."
                    ),
                },
                {"role": "user", "content": text},
            ],
            "stream": False,
        }

        try:
            response = requests.post(f"{OLLAMA_URL}/api/chat", json=payload, timeout=120)
            if response.status_code == 200:
                res_data = response.json()
                reply = res_data.get("message", {}).get("content", "").strip()
                logging.info(f"LLM Response: '{reply}'")
                return reply
            else:
                err_msg = f"Ollama HTTP {response.status_code}: {response.text}"
                logging.error(err_msg)
                return "I encountered an error communicating with my local LLM model."
        except requests.exceptions.Timeout:
            logging.error("Ollama request timed out after 120 seconds (first query cold-start or slow response).")
            return "The local AI model took too long to respond. Please ask your question again."
        except requests.exceptions.ConnectionError:
            logging.error(f"Failed to connect to Ollama at {OLLAMA_URL}. Ensure Ollama service is running.")
            return "I cannot connect to Ollama. Please make sure the Ollama application is running on your PC."
        except Exception as e:
            logging.error(f"Unexpected error calling Ollama API: {e}")
            return "An unexpected error occurred while generating a response."

    def speak(self, text):
        """Convert response text to spoken audio using pyttsx3."""
        if not text:
            return

        self.set_state("speaking", "Speaking response...")
        logging.info(f"Speaking: '{text}'")

        try:
            # Initialize pyttsx3 inside thread with SAPI5 COM setup
            try:
                import pythoncom
                pythoncom.CoInitialize()
            except ImportError:
                pass

            engine = pyttsx3.init("sapi5" if os.name == "nt" else None)
            engine.setProperty("rate", 175)  # Natural speaking rate
            voices = engine.getProperty("voices")
            if voices:
                engine.setProperty("voice", voices[0].id)

            engine.say(text)
            engine.runAndWait()
            engine.stop()
        except Exception as e:
            logging.error(f"Text-to-speech failed: {e}")

    def run_assistant_loop(self):
        """Main background thread loop for Jarvis voice interactions."""
        # Initialize whisper model lazily in background thread
        self.whisper_model = self.init_whisper()
        if not self.whisper_model:
            self.set_state("error", "Whisper STT failed to load")
            return

        # Initialize openWakeWord engine
        self.oww_model = self.init_openwakeword()
        if not self.oww_model:
            self.set_state("error", "openWakeWord failed to load")
            return

        dev_idx = INPUT_DEVICE_INDEX if INPUT_DEVICE_INDEX is not None else sd.default.device[0]
        try:
            device_info = sd.query_devices(dev_idx, "input")
            dev_channels = max(1, device_info["max_input_channels"])
            logging.info(f"Using audio input device [{dev_idx}]: '{device_info['name']}' ({dev_channels} channels)")
        except Exception as dev_err:
            dev_channels = 1
            logging.warning(f"Could not query input device info: {dev_err}")

        self.set_state("listening", f"Listening for '{WAKE_WORD_MODEL}'")

        silent_chunk_count = 0
        warned_silence = False

        try:
            with sd.RawInputStream(
                samplerate=SAMPLE_RATE,
                blocksize=CHUNK_SIZE,
                dtype="int16",
                channels=dev_channels,
                device=INPUT_DEVICE_INDEX,
            ) as stream:
                while not self.stop_event.is_set():
                    if self.is_paused:
                        time.sleep(0.5)
                        continue

                    # Read chunk for openWakeWord detection
                    pcm_data, overflowed = stream.read(CHUNK_SIZE)
                    if overflowed:
                        continue

                    pcm_raw = np.frombuffer(pcm_data, dtype=np.int16)
                    if dev_channels > 1 and len(pcm_raw) >= dev_channels:
                        pcm_matrix = pcm_raw.reshape(-1, dev_channels)
                        pcm = np.mean(pcm_matrix, axis=1).astype(np.float32)
                    else:
                        pcm = pcm_raw.astype(np.float32)

                    # Apply digital software gain for low-gain mic hardware
                    pcm = np.clip(pcm * AUDIO_GAIN, -32768, 32767).astype(np.int16)

                    # Diagnostic check for dead/muted microphone signal
                    max_amp = np.max(np.abs(pcm)) if len(pcm) > 0 else 0
                    if max_amp <= 2:
                        silent_chunk_count += 1
                        if silent_chunk_count >= 100 and not warned_silence:
                            logging.warning(
                                "Microphone signal level is near ZERO (Max Amp <= 2). "
                                "Your microphone may be muted, disabled, or set to the wrong input device. "
                                "Run 'python jarvis.py --list-devices' to check available microphones."
                            )
                            warned_silence = True
                    else:
                        silent_chunk_count = 0
                        warned_silence = False

                    prediction = self.oww_model.predict(pcm)
                    score = prediction.get(WAKE_WORD_MODEL, 0.0)

                    if score >= WAKE_WORD_THRESHOLD:
                        logging.info(f">>> WAKE WORD DETECTED ('{WAKE_WORD_MODEL}', score: {score:.2f}) <<<")
                        self.oww_model.reset()

                        # 1. Record Speech Input
                        audio_data = self.record_audio()
                        if audio_data is None or len(audio_data) == 0:
                            logging.info("No speech detected after wake word.")
                            self.set_state("listening", f"Listening for '{WAKE_WORD_MODEL}'")
                            continue

                        # 2. Transcribe Audio
                        user_text = self.transcribe_audio(audio_data)
                        if not user_text:
                            self.speak("I couldn't understand what you said.")
                            self.set_state("listening", f"Listening for '{WAKE_WORD_MODEL}'")
                            continue

                        if self.on_user_transcript:
                            self.on_user_transcript(user_text)

                        # 3. Get LLM Response
                        llm_reply = self.get_llm_response(user_text)

                        if self.on_llm_response:
                            self.on_llm_response(llm_reply)

                        # 4. Speak Response
                        self.speak(llm_reply)

                        # 5. Reset to wake word listening
                        self.set_state("listening", f"Listening for '{WAKE_WORD_MODEL}'")

        except Exception as e:
            logging.error(f"Fatal error in assistant loop: {e}", exc_info=True)
            self.set_state("error", f"Error: {e}")
        finally:
            logging.info("Jarvis Assistant loop terminated.")

    def process_text_prompt(self, user_text):
        """Process a typed text prompt from GUI or API directly."""
        if not user_text or self.is_paused:
            return
        threading.Thread(target=self._exec_text_prompt, args=(user_text,), daemon=True).start()

    def _exec_text_prompt(self, user_text):
        if self.on_user_transcript:
            self.on_user_transcript(user_text)
        llm_reply = self.get_llm_response(user_text)
        if self.on_llm_response:
            self.on_llm_response(llm_reply)
        self.speak(llm_reply)
        self.set_state("listening", f"Listening for '{WAKE_WORD_MODEL}'")

    def toggle_pause(self, icon, item):
        """Toggle pause state from system tray menu."""
        self.is_paused = not self.is_paused
        if self.is_paused:
            self.set_state("paused", "Paused")
        else:
            self.set_state("listening", f"Listening for '{WAKE_WORD_MODEL}'")

    def quit_app(self, icon, item):
        """Stop background worker and terminate tray icon."""
        logging.info("Shutdown request received from System Tray menu.")
        self.stop_event.set()
        if self.tray_icon:
            self.tray_icon.stop()

    def run_tray_icon(self):
        """Run system tray icon loop on the main thread."""
        menu = pystray.Menu(
            pystray.MenuItem("Jarvis Voice Assistant", None, enabled=False),
            pystray.Menu.SEPARATOR,
            pystray.MenuItem(lambda item: f"Status: {self.state_text}", None, enabled=False),
            pystray.MenuItem(
                lambda item: "Resume Listening" if self.is_paused else "Pause Listening",
                self.toggle_pause,
            ),
            pystray.Menu.SEPARATOR,
            pystray.MenuItem("Quit Jarvis", self.quit_app),
        )

        initial_image = create_tray_icon_image("listening")
        self.tray_icon = pystray.Icon("jarvis_assistant", initial_image, "Jarvis Assistant", menu)

        # Start assistant worker thread
        worker_thread = threading.Thread(target=self.run_assistant_loop, daemon=True)
        worker_thread.start()

        # Run pystray main GUI loop (blocking main thread)
        logging.info("Starting System Tray Icon loop...")
        self.tray_icon.run()
        logging.info("System Tray Icon loop ended. Jarvis exiting.")


def list_audio_devices():
    """Utility to list all available audio input devices."""
    print("\nAvailable Audio Input Devices:")
    print("-----------------------------------------------------------------------------")
    devices = sd.query_devices()
    default_in = sd.default.device[0]
    for idx, dev in enumerate(devices):
        if dev["max_input_channels"] > 0:
            host = sd.query_hostapis(dev["hostapi"])["name"]
            is_default = " (DEFAULT)" if idx == default_in else ""
            print(f"[{idx:2d}] {dev['name']:40s} | Host: {host:15s} | Channels: {dev['max_input_channels']}{is_default}")
    print("-----------------------------------------------------------------------------\n")


# ==========================================
# MAIN ENTRYPOINT
# ==========================================
def main():
    if "--list-devices" in sys.argv:
        list_audio_devices()
        return

    setup_logging()
    assistant = JarvisAssistant()
    assistant.run_tray_icon()


if __name__ == "__main__":
    main()

