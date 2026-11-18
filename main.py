import os, gc, uuid, shutil, zipfile, traceback, subprocess, cv2
from faster_whisper import WhisperModel
from fastapi import FastAPI, UploadFile, File, Form
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

app = FastAPI()
os.makedirs("uploads", exist_ok=True)
os.makedirs("outputs", exist_ok=True)
app.mount("/outputs", StaticFiles(directory="outputs"), name="outputs")
model = WhisperModel("tiny", device="cpu", compute_type="int8")

def get_face_center_x(video_path: str, start_sec: float, duration: float) -> int:
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    cap.set(cv2.CAP_PROP_POS_FRAMES, int(start_sec * fps))
    centers = []
    for _ in range(int(duration)):
        ret, frame = cap.read()
        if not ret: break
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.1, 4)
        if len(faces) > 0:
            x, y, w, h = max(faces, key=lambda f: f[2] * f[3])
            centers.append(x + w // 2)
        cap.set(cv2.CAP_PROP_POS_FRAMES, cap.get(cv2.CAP_PROP_POS_FRAMES) + int(fps))
    cap.release()
    return int(sum(centers) / len(centers)) if centers else width // 2

def generate_animated_ass(words: list, start_offset: float, ass_path: str):
    header = "[Script Info]\nScriptType: v4.00+\nPlayResX: 1080\nPlayResY: 1920\n\n[V4+ Styles]\nFormat: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding\nStyle: Default,Arial,85,&H00FFFFFF,&H0000FFFF,&H00000000,&H80000000,-1,0,0,0,100,100,0,0,1,5,2,2,50,50,260,1\n\n[Events]\nFormat: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text\n"
    def fmt_time(s: float) -> str:
        return f"{int(s//3600)}:{int((s%3600)//60):02d}:{s%60:05.2f}"
    events = []
    for i in range(0, len(words), 3):
        chunk = words[i:i+3]
        if not chunk: continue
        c_start = max(0, chunk[0]["start"] - start_offset)
        c_end = max(c_start + 0.2, chunk[-1]["end"] - start_offset)
        text = " ".join([w["word"].strip().upper() for w in chunk])
        anim = r"{\fscx80\fscy80\t(0,80,\fscx105\fscy105)\t(80,150,\fscx100\fscy100)\c&H0000FFFF&}"
        events.append(f"Dialogue: 0,{fmt_time(c_start)},{fmt_time(c_end)},Default,,0,0,0,,{anim}{text}")
    with open(ass_path, "w", encoding="utf-8") as f:
        f.write(header + "\n".join(events))

def detect_highlights(segments: list, clip_duration: int = 30) -> list:
    hooks = {"why", "how", "secret", "never", "stop", "mistake", "truth", "best", "worst", "money", "ai"}
    scored = []
    for seg in segments:
        start = seg["start"]
        win = [s for s in segments if s["start"] >= start and s["end"] <= start + clip_duration]
        if not win: continue
        txt = " ".join(s["text"].lower() for s in win)
        score = len(txt.split()) + sum(10 for w in hooks if w in txt) + txt.count("?") * 15
        scored.append((score, start, win))
    scored.sort(key=lambda x: x[0], reverse=True)
    selected = []
    for score, start, win in scored:
        if all(abs(start - s[0]) >= clip_duration for s in selected):
            selected.append((start, win))
        if len(selected) == 3: break
    return selected

@app.get("/")
def serve_ui():
    return FileResponse("static/index.html")

@app.post("/api/process")
async def process_video(file: UploadFile = File(...), aspect_ratio: str = Form("auto"), ai_style: str = Form("mixed")):
    try:
        job_id = str(uuid.uuid4())[:8]
        job_dir = os.path.join("outputs", job_id)
        os.makedirs(job_dir, exist_ok=True)
        input_path = os.path.join("uploads", f"{job_id}_{file.filename}")
        with open(input_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        raw_segs, _ = model.transcribe(input_path, word_timestamps=True)
        segments = [{"start": s.start, "end": s.end, "text": s.text, "words": [{"word": w.word, "start": w.start, "end": w.end} for w in (s.words or [])]} for s in raw_segs]
        highlights = detect_highlights(segments, 30) or [(0.0, segments)]
        cap = cv2.VideoCapture(input_path)
        orig_w, orig_h = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        cap.release()
        clip_urls, clip_paths = [], []
        for idx, (start_time, win_segs) in enumerate(highlights):
            words = [w for s in win_segs for w in s.get("words", [])]
            ass_path = os.path.join(job_dir, f"subs_{idx}.ass")
            generate_animated_ass(words, start_time, ass_path)
            if aspect_ratio in ("auto", "9:16"):
                face_x = get_face_center_x(input_path, start_time, 30)
                target_w = min(int(orig_h * (9 / 16)), orig_w)
                crop_x = max(0, min(orig_w - target_w, face_x - target_w // 2))
                vf = f"crop={target_w}:{orig_h}:{crop_x}:0,scale=1080:1920,ass={ass_path}"
            else:
                vf = f"scale=1920:1080,ass={ass_path}"
            out_clip = os.path.join(job_dir, f"clip_{idx + 1}.mp4")
            subprocess.run(["ffmpeg", "-y", "-threads", "1", "-ss", str(start_time), "-t", "30", "-i", input_path, "-vf", vf, "-c:v", "libx264", "-preset", "ultrafast", "-crf", "26", "-c:a", "aac", "-b:a", "128k", out_clip], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            clip_urls.append(f"/outputs/{job_id}/clip_{idx + 1}.mp4")
            clip_paths.append(out_clip)
        zip_name = f"clips_{job_id}.zip"
        with zipfile.ZipFile(os.path.join(job_dir, zip_name), "w") as zipf:
            for p in clip_paths:
                if os.path.exists(p): zipf.write(p, os.path.basename(p))
        if os.path.exists(input_path): os.remove(input_path)
        gc.collect()
        return JSONResponse({"job_id": job_id, "clips": clip_urls, "zip_url": f"/outputs/{job_id}/{zip_name}"})
    except Exception as e:
        traceback.print_exc()
        return JSONResponse({"error": str(e)}, status_code=500)
