# Jarvis - Windows Desktop Voice Assistant (Phase 1)

**Jarvis** is a lightweight, local-first Python desktop voice assistant that runs silently in the Windows System Tray (similar to Discord or Spotify).

It listens continuously for the wake word **"Hey Jarvis"** using **openWakeWord**, records user speech input, transcribes it locally using **Faster-Whisper** (GPU/CUDA accelerated), queries a local LLM via **Ollama** (`llama3.1:8b`), and speaks the response back using offline speech synthesis (**pyttsx3**).

---

## 🌟 Key Features

- 🤫 **Silent System Tray Operation**: No visible command prompt window when packaged. Runs quietly near the Windows clock.
- 🎨 **Dynamic Visual State Indicator**: System tray icon dynamically changes colors to show assistant state:
  - 🔵 **Cyan Circle**: Idle / Listening for wake word (`"hey_jarvis"`)
  - 🟡 **Amber Pulse**: Recording speech or waiting for Ollama LLM
  - 🟣 **Purple Glow**: Speaking response out loud
  - ⚪ **Muted Grey**: Paused
- 🔓 **100% Free & Open Source**: Powered by **openWakeWord** — no API keys, accounts, or cloud signups required.
- ⚡ **Low CPU Idle**: Continuous background wake word monitoring.
- 🚀 **Local & Private**: All wake word detection (openWakeWord), transcription (Faster-Whisper), and LLM inference (Ollama `llama3.1:8b`) run 100% locally on your machine.
- 📝 **File Logging**: Logs state transitions, speech transcripts, and diagnostics silently to `jarvis.log`.

---

## 📋 System Requirements

- **Operating System**: Windows 10 or Windows 11 (64-bit)
- **Python**: Version 3.10 or 3.11 recommended
- **GPU**: NVIDIA GPU with CUDA support (falls back to CPU if CUDA is unavailable)
- **Ollama**: Installed and running locally on Windows ([Download Ollama](https://ollama.com/))
- **No API Keys Needed!**

---

## 🚀 First-Time Setup Instructions

### 1. Install & Configure Ollama
Make sure Ollama is installed and running, then pull the `llama3.1:8b` model in your terminal:

```powershell
ollama pull llama3.1:8b
```

Verify Ollama is active by visiting `http://localhost:11434` in your browser.

### 2. Install Python Dependencies
In PowerShell, navigate to the project folder and run:

```powershell
pip install -r requirements.txt
```

*(openWakeWord will automatically download the pre-trained `hey_jarvis` ONNX model on its first run.)*

---

## 🧪 Testing Before Packaging

Run Jarvis directly in Python to test all components:

```powershell
python jarvis.py
```

### What to expect:
1. A new icon will appear in your system tray.
2. Check `jarvis.log` or console output for:
   ```text
   Jarvis Assistant Initializing...
   Faster-Whisper loaded successfully on CUDA GPU.
   openWakeWord loaded model 'hey_jarvis'.
   State -> LISTENING: Listening for 'hey_jarvis'
   ```
3. Say **"Hey Jarvis"** clearly into your microphone.
4. Speak your prompt (e.g., *"What is the capital of France?"*).
5. Listen to Jarvis respond out loud.
6. Right-click the system tray icon to check status or click **Quit Jarvis**.

---

## 📦 Packaging to `.exe` & Auto-Start at Login

To package Jarvis into a standalone executable that runs automatically at Windows startup without any visible terminal windows, follow the instructions in [`build.md`](file:///d:/personal_assistant/build.md).

Quick PyInstaller command:

```powershell
pyinstaller --noconsole --onefile --name Jarvis `
  --collect-all openwakeword `
  --collect-all onnxruntime `
  --collect-all faster_whisper `
  --collect-all ctranslate2 `
  --collect-all pyttsx3 `
  --collect-all pystray `
  jarvis.py
```

---

## 📁 File Structure

- [`jarvis.py`](file:///d:/personal_assistant/jarvis.py): Main assistant application & system tray logic.
- [`requirements.txt`](file:///d:/personal_assistant/requirements.txt): Python dependency definitions.
- [`build.md`](file:///d:/personal_assistant/build.md): Executable build and Windows Startup instructions.
- [`README.md`](file:///d:/personal_assistant/README.md): Project setup guide.
- `jarvis.log`: Generated runtime log file.
