import os
import gc
import uuid
import shutil
import zipfile
import subprocess
import cv2
from faster_whisper import WhisperModel
from fastapi import FastAPI, UploadFile, File, Form
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

app = FastAPI()

os.makedirs("uploads", exist_ok=True)
os.makedirs("outputs", exist_ok=True)
os.makedirs("static", exist_ok=True)

app.mount("/static", StaticFiles(directory="static"), name="static")
app.mount("/outputs", StaticFiles(directory="outputs"), name="outputs")

# Load lightweight INT8 quantized Whisper model (~120MB RAM total)
model = WhisperModel("tiny", device="cpu", compute_type="int8")


def get_face_center_x(video_path: str, start_sec: float, duration: float) -> int:
    """Uses OpenCV Haar Cascades to find the average horizontal center of a speaker's face."""
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
            x, y, w, h = max(faces, key=lambda f: f * f)
            centers.append(x + w // 2)
        cap.set(cv2.CAP_PROP_POS_FRAMES, cap.get(cv2.CAP_PROP_POS_FRAMES) + int(fps))
        
    cap.release()
    return int(sum(centers) / len(centers)) if centers else width // 2


def generate_animated_ass(words: list, start_offset: float, ass_path: str):
    """Creates an Advanced SubStation Alpha (.ass) file with pop-in animated captions."""
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
        c_start = max(0, chunk["start"] - start_offset)
        c_end = max(c_start + 0.2, chunk[-1]["end"] - start_offset)
        
        text = " ".join([w["word"].strip().upper() for w in chunk])
        anim_tag = r"{\fscx80\fscy80\t(0,80,\fscx105\fscy105)\t(80,150,\fscx100\fscy100)\c&H0000FFFF&}"
        events.append(f"Dialogue: 0,{fmt_time(c_start)},{fmt_time(c_end)},Default,,0,0,0,,{anim_tag}{text}")

    with open(ass_path, "w", encoding="utf-8") as f:
        f.write(header + "\n".join(events))


def detect_highlights(segments: list, clip_duration: int = 30) -> list:
    """Scores transcript segments for high-energy hooks and questions."""
    hook_words = {"why", "how", "secret", "never", "stop", "mistake", "truth", "best", "worst", "money", "ai"}
    scored_windows = []

    for seg in segments:
        start = seg["start"]
        window_segs = [s for s in segments if s["start"] >= start and s["end"] <= start + clip_duration]
        if not window_segs:
            continue
            
        full_text = " ".join(s["text"].lower() for s in window_segs)
        score = len(full_text.split())
        score += sum(10 for w in hook_words if w in full_text)
        score += full_text.count("?") * 15
        
        scored_windows.append((score, start, window_segs))

    scored_windows.sort(key=lambda x: x, reverse=True)
    selected = []
    for score, start, window_segs in scored_windows:
        if all(abs(start - s) >= clip_duration for s in selected):
            selected.append((start, window_segs))
        if len(selected) == 3:
            break
    return selected


@app.get("/")
def serve_ui():
    return FileResponse("static/index.html")


@app.post("/api/process")
async def process_video(
    file: UploadFile = File(...),
    aspect_ratio: str = Form("auto"),
    ai_style: str = Form("mixed")
):
    job_id = str(uuid.uuid4())[:8]
    job_dir = os.path.join("outputs", job_id)
    os.makedirs(job_dir, exist_ok=True)

    # Stream file to disk in chunks to prevent RAM spikes
    input_path = os.path.join("uploads", f"{job_id}_{file.filename}")
    with open(input_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    # 1. Transcribe with faster-whisper
    raw_segments, _ = model.transcribe(input_path, word_timestamps=True)
    segments = []
    for seg in raw_segments:
        words_list = [{"word": w.word, "start": w.start, "end": w.end} for w in (seg.words or [])]
        segments.append({
            "start": seg.start,
            "end": seg.end,
            "text": seg.text,
            "words": words_list
        })

    # 2. Detect top 30-second highlights
    highlights = detect_highlights(segments, clip_duration=30)
    if not highlights:
        highlights = [(0.0, segments)]

    cap = cv2.VideoCapture(input_path)
    orig_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    orig_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()

    clip_urls = []
    clip_paths = []

    for idx, (start_time, window_segs) in enumerate(highlights):
        words = []
        for s in window_segs:
            words.extend(s.get("words", []))

        ass_path = os.path.join(job_dir, f"subs_{idx}.ass")
        generate_animated_ass(words, start_time, ass_path)

        # 3. Aspect Ratio & OpenCV Face Tracking Crop
        if aspect_ratio == "auto" or aspect_ratio == "9:16":
            face_x = get_face_center_x(input_path, start_time, 30)
            target_w = int(orig_h * (9 / 16))
            target_w = min(target_w, orig_w)
            crop_x = max(0, min(orig_w - target_w, face_x - target_w // 2))
            vf_filter = f"crop={target_w}:{orig_h}:{crop_x}:0,scale=1080:1920,ass={ass_path}"
        else:
            vf_filter = f"scale=1920:1080,ass={ass_path}"

        out_clip = os.path.join(job_dir, f"clip_{idx + 1}.mp4")
        
        # 4. FFmpeg Cut, Crop & Burn Animated Captions (single thread to save RAM)
        cmd = [
            "ffmpeg", "-y",
            "-threads", "1",
            "-ss", str(start_time),
            "-t", "30",
            "-i", input_path,
            "-vf", vf_filter,
            "-c:v", "libx264", "-preset", "ultrafast", "-crf", "26",
            "-c:a", "aac", "-b:a", "128k",
            out_clip
        ]
        subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        
        clip_urls.append(f"/outputs/{job_id}/clip_{idx + 1}.mp4")
        clip_paths.append(out_clip)

    # 5. Bundle all clips into a single ZIP file
    zip_filename = f"clips_{job_id}.zip"
    zip_path = os.path.join(job_dir, zip_filename)
    with zipfile.ZipFile(zip_path, "w") as zipf:
        for path in clip_paths:
            if os.path.exists(path):
                zipf.write(path, os.path.basename(path))

    # Clean up uploaded raw file and free memory
    if os.path.exists(input_path):
        os.remove(input_path)
    gc.collect()

    return JSONResponse({
        "job_id": job_id,
        "clips": clip_urls,
        "zip_url": f"/outputs/{job_id}/{zip_filename}"
    })
