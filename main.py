import os
import gc
import uuid
import shutil
import zipfile
import traceback
import subprocess
import cv2
from faster_whisper import WhisperModel
from fastapi import FastAPI, UploadFile, File, Form
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

app = FastAPI()

os.makedirs("uploads", exist_ok=True)
os.makedirs("outputs", exist_ok=True)

app.mount("/outputs", StaticFiles(directory="outputs"), name="outputs")

# Load lightweight INT8 quantized Whisper model (~120MB RAM total)
model = WhisperModel("tiny", device="cpu", compute_type="int8")

HTML_UI = """<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1.0" />
  <title>NoxClips AI — Free Open-Source Clipper</title>
  <style>
    :root {
      --bg: #0b0d12;
      --surface: #141821;
      --border: #232938;
      --accent: #6366f1;
      --accent-hover: #4f46e5;
      --text: #f3f4f6;
      --muted: #9ca3af;
    }
    * { box-sizing: border-box; font-family: 'Inter', system-ui, sans-serif; }
    body {
      background: var(--bg);
      color: var(--text);
      margin: 0;
      padding: 2rem;
      display: flex;
      flex-direction: column;
      align-items: center;
    }
    .container { max-width: 900px; width: 100%; }
    header { text-align: center; margin-bottom: 2rem; }
    header h1 { font-size: 2rem; margin: 0 0 0.5rem; }
    header p { color: var(--muted); margin: 0; }

    .card {
      background: var(--surface);
      border: 1px solid var(--border);
      border-radius: 14px;
      padding: 1.5rem;
      margin-bottom: 1.5rem;
    }
    .settings-grid {
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
      gap: 1rem;
      margin-bottom: 1.5rem;
    }
    .badge-box {
      background: #0f1219;
      border: 1px solid var(--border);
      padding: 0.75rem 1rem;
      border-radius: 10px;
    }
    .badge-box span { display: block; font-size: 0.75rem; color: var(--muted); }
    .badge-box strong { font-size: 0.95rem; color: #a5b4fc; }

    .upload-zone {
      border: 2px dashed var(--border);
      border-radius: 12px;
      padding: 2.5rem;
      text-align: center;
      cursor: pointer;
      transition: border-color 0.2s;
      display: block;
    }
    .upload-zone:hover { border-color: var(--accent); }
    input[type="file"] { display: none; }

    .btn {
      background: var(--accent);
      color: white;
      border: none;
      padding: 0.85rem 1.5rem;
      border-radius: 10px;
      font-weight: 600;
      cursor: pointer;
      width: 100%;
      font-size: 1rem;
      margin-top: 1rem;
    }
    .btn:hover { background: var(--accent-hover); }
    .btn:disabled { opacity: 0.5; cursor: not-allowed; }
    .btn-zip { background: #10b981; margin-bottom: 1.5rem; display: inline-block; text-align: center; text-decoration: none; }
    .btn-zip:hover { background: #059669; }

    .clips-grid {
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
      gap: 1.25rem;
    }
    .clip-card {
      background: #0f1219;
      border: 1px solid var(--border);
      border-radius: 12px;
      padding: 1rem;
      text-align: center;
    }
    video { width: 100%; border-radius: 8px; background: #000; margin-bottom: 0.75rem; }
    .dl-link {
      display: block;
      padding: 0.6rem;
      background: var(--surface);
      border: 1px solid var(--border);
      color: var(--text);
      border-radius: 8px;
      text-decoration: none;
      font-size: 0.9rem;
    }
    .dl-link:hover { border-color: var(--accent); }
    #status { text-align: center; margin-top: 1rem; color: #a5b4fc; }
  </style>
</head>
<body>
  <div class="container">
    <header>
      <h1>⚡ NoxClips AI</h1>
      <p>Whisper + OpenCV Face Tracking + FFmpeg Animated Captions</p>
    </header>

    <div class="card">
      <div class="settings-grid">
        <div class="badge-box"><span>Duration</span><strong>30 Seconds</strong></div>
        <div class="badge-box"><span>AI Style</span><strong>Mixed (Hooks + Density)</strong></div>
        <div class="badge-box"><span>Aspect Ratio</span><strong>Auto (Face Track 9:16)</strong></div>
        <div class="badge-box"><span>Captions</span><strong>Animated Pop-In</strong></div>
      </div>

      <label class="upload-zone" for="videoInput">
        <div id="fileLabel">📁 Click to select a video file (MP4, MOV, MKV)</div>
        <input type="file" id="videoInput" accept="video/*" />
      </label>

      <button class="btn" id="generateBtn" disabled>Generate 30s Clips</button>
      <div id="status"></div>
    </div>

    <div id="resultsSection" style="display: none;">
      <a id="zipBtn" class="btn btn-zip" href="#" download>📦 Download All Clips (ZIP)</a>
      <div class="clips-grid" id="clipsGrid"></div>
    </div>
  </div>

  <script>
    const videoInput = document.getElementById('videoInput');
    const fileLabel = document.getElementById('fileLabel');
    const generateBtn = document.getElementById('generateBtn');
    const status = document.getElementById('status');
    const resultsSection = document.getElementById('resultsSection');
    const clipsGrid = document.getElementById('clipsGrid');
    const zipBtn = document.getElementById('zipBtn');

    videoInput.addEventListener('change', () => {
      if (videoInput.files.length > 0) {
        fileLabel.textContent = `Selected: ${videoInput.files[0].name}`;
        generateBtn.disabled = false;
      }
    });

    generateBtn.addEventListener('click', async () => {
      const file = videoInput.files[0];
      if (!file) return;

      generateBtn.disabled = true;
      resultsSection.style.display = 'none';
      status.textContent = '⏳ Transcribing with Whisper, tracking faces, and rendering captions...';

      const formData = new FormData();
      formData.append('file', file);
      formData.append('aspect_ratio', 'auto');
      formData.append('ai_style', 'mixed');

      try {
        const res = await fetch('/api/process', { method: 'POST', body: formData });
        const data = await res.json();

        if (!res.ok) {
          throw new Error(data.error || 'Server processing failed');
        }

        clipsGrid.innerHTML = '';
        data.clips.forEach((url, index) => {
          const card = document.createElement('div');
          card.className = 'clip-card';
          card.innerHTML = `
            <video src="${url}" controls></video>
            <a class="dl-link" href="${url}" download="clip_${index + 1}.mp4">⬇️ Download Clip ${index + 1}</a>
          `;
          clipsGrid.appendChild(card);
        });

        zipBtn.href = data.zip_url;
        resultsSection.style.display = 'block';
        status.textContent = '✅ Done! Your clips are ready.';
      } catch (err) {
        status.textContent = `❌ Error: ${err.message}`;
      } finally {
        generateBtn.disabled = false;
      }
    });
  </script>
</body>
</html>
"""


def get_face_center_x(video_path: str, start_sec: float, duration: float) -> int:
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    
    cap.set(cv2.CAP_PROP_POS_FRAMES, int(start_sec * fps))
    centers = []
    
    for _ in range(int(duration)):
        ret, frame = cap.read()
        if not ret:
            break
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.1, 4)
        if len(faces) > 0:
            x, y, w, h = max(faces, key=lambda f: f[2] * f[3])
            centers.append(x + w // 2)
        cap.set(cv2.CAP_PROP_POS_FRAMES, cap.get(cv2.CAP_PROP_POS_FRAMES) + int(fps))
        
    cap.release()
    return int(sum(centers) / len(centers)) if centers else width // 2


def generate_animated_ass(words: list, start_offset: float, ass_path: str):
    header = """[Script Info]
ScriptType: v4.00+
PlayResX: 1080
PlayResY: 1920

[V4+ Styles]
Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding
Style: Default,Arial,85,&H00FFFFFF,&H0000FFFF,&H00000000,&H80000000,-1,0,0,0,100,100,0,0,1,5,2,2,50,50,260,1

[Events]
Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text
"""
    def fmt_time(seconds: float) -> str:
        hrs = int(seconds // 3600)
        mins = int((seconds % 3600) // 60)
        secs = seconds % 60
        return f"{hrs}:{mins:02d}:{secs:05.2f}"

    events = []
    chunk_size = 3
    for i in range(0, len(words), chunk_size):
        chunk = words[i:i + chunk_size]
        if not chunk:
            continue
        c_start = max(0, chunk[0]["start"] - start_offset)
        c_end = max(c_start + 0.2, chunk[-1]["end"] - start_offset)
        
        text = " ".join([w["word"].strip().upper() for w in chunk])
        anim_tag = r"{\fscx80\fscy80\t(0,80,\fscx105\fscy105)\t(80,150,\fscx100\fscy100)\c&H0000FFFF&}"
        events.append(f"Dialogue: 0,{fmt_time(c_start)},{fmt_time(c_end)},Default,,0,0,0,,{anim_tag}{text}")

    with open(ass_path, "w", encoding="utf-8") as f:
        f.write(header + "\n".join(events))


def detect_highlights(segments: list, clip_duration: int = 30) -> list:
    hook_words = {"why", "how", "secret", "never", "stop", "mistake", "truth", "best", "worst", "money", "ai"}
    scored_windows = []

    for seg in segments:
        start = seg["start"]
        window_segs = [s for s in segments if s["start"] >= start and s["end"] <= start + clip_duration]
        if not window_segs:
            continue
            
        full_text = " ".join(s["text"].lower() for s in window_segs)
