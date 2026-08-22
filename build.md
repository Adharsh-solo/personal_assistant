# Building & Deploying Jarvis Standalone Executable (.exe)

This guide walks you through packaging **Jarvis** into a single, windowless `.exe` file using PyInstaller and configuring it to run silently on Windows Startup.

---

## Step 1: Prerequisites

Ensure all required dependencies are installed in your Python environment:

```powershell
pip install -r requirements.txt
```

Verify that PyInstaller is installed:

```powershell
pyinstaller --version
```

*Note: No API keys, signups, or accounts are required. Wake word detection runs 100% locally via openWakeWord.*

---

## Step 2: Package with PyInstaller

Run the following command in PowerShell inside your project directory (`d:\personal_assistant`):

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

### Explanation of Flags:
- `--noconsole` (`--windowed`): Suppresses the command prompt / console window, making Jarvis run silently in the background.
- `--onefile`: Bundles all Python code, standard libraries, and dependencies into a single `Jarvis.exe` file.
- `--name Jarvis`: Sets the binary output name to `Jarvis.exe`.
- `--collect-all <package>`: Ensures model files, ONNX runtime binaries, and platform drivers for `openwakeword`, `onnxruntime`, `faster_whisper`, `ctranslate2`, `pyttsx3`, and `pystray` are bundled cleanly into the executable.

After the build finishes (usually 1–2 minutes), the output executable will be created at:
`d:\personal_assistant\dist\Jarvis.exe`

---

## Step 3: Auto-Launch Jarvis on Windows Login (Silent Startup)

To have Jarvis start automatically in the background whenever you log into Windows:

### Option A: Manual Shortcut Setup
1. Press `Win + R` to open the **Run** dialog.
2. Type `shell:startup` and press **Enter**. This opens your personal Windows Startup folder (`%APPDATA%\Microsoft\Windows\Start Menu\Programs\Startup`).
3. Right-click inside the folder and select **New > Shortcut**.
4. Click **Browse...** and select `d:\personal_assistant\dist\Jarvis.exe`.
5. Click **Next**, name the shortcut `Jarvis`, and click **Finish**.

### Option B: Automated PowerShell Setup
Alternatively, run this single line in PowerShell to create the shortcut automatically:

```powershell
$WshShell = New-Object -ComObject WScript.Shell; $Shortcut = $WshShell.CreateShortcut("$env:APPDATA\Microsoft\Windows\Start Menu\Programs\Startup\Jarvis.lnk"); $Shortcut.TargetPath = (Get-Item "dist\Jarvis.exe").FullName; $Shortcut.WorkingDirectory = (Get-Item "dist").FullName; $Shortcut.Save()
```

---

## Step 4: Verification & Background Operation

1. Double-click `dist\Jarvis.exe` (or restart your computer to test auto-startup).
2. Look at the Windows system tray (bottom-right near the clock, click the `^` hidden icons arrow if needed).
3. You will see the **Jarvis system tray icon** (a dark slate circle with a glowing cyan dot when idle/listening).
4. Say **"Hey Jarvis"** clearly into your microphone:
   - The tray icon will pulse **amber** while recording and processing.
   - Jarvis will generate an answer via Ollama.
   - The tray icon will pulse **purple** while speaking the response out loud.
   - The tray icon will return to **cyan** (idle/listening).
5. Right-click the system tray icon to view status, pause/resume, or click **Quit Jarvis** to close it.
6. Check `jarvis.log` in the directory where `Jarvis.exe` is located for real-time background logs.
