import os, gc, uuid, shutil, zipfile, traceback, subprocess, cv2
from faster_whisper import WhisperModel
from fastapi import FastAPI, UploadFile, File, Form
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

app = FastAPI()
os.makedirs("uploads", exist_ok=True)
os.makedirs("outputs", exist_ok=True)
app.mount("/outputs", StaticFiles(directory="outputs"), name="outputs")
model = WhisperModel("tiny", device="cpu", compute_type="int8")

UI = """<!DOCTYPE html><html><head><meta charset="UTF-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>NoxClips AI</title><style>body{background:#0b0d12;color:#f3f4f6;font-family:system-ui,sans-serif;margin:0;padding:1.5rem;display:flex;justify-content:center}.box{max-width:750px;width:100%}header{text-align:center;margin-bottom:1.5rem}.card{background:#141821;border:1px solid #232938;border-radius:14px;padding:1.25rem;margin-bottom:1.5rem}.grid{display:grid;grid-template-columns:repeat(2,1fr);gap:.75rem;margin-bottom:1.25rem}.b{background:#0f1219;border:1px solid #232938;padding:.65rem;border-radius:8px}.b span{display:block;font-size:.7rem;color:#9ca3af}.b strong{font-size:.85rem;color:#a5b4fc}.up{border:2px dashed #232938;border-radius:12px;padding:2rem 1rem;text-align:center;cursor:pointer;display:block}input{display:none}.btn{background:#6366f1;color:#fff;border:none;padding:.85rem;border-radius:10px;font-weight:600;cursor:pointer;width:100%;font-size:1rem;margin-top:1rem}.btn:disabled{opacity:.5}.zip{background:#10b981;display:block;text-align:center;text-decoration:none;margin-bottom:1rem}.clips{display:grid;grid-template-columns:repeat(auto-fit,minmax(220px,1fr));gap:1rem}.cc{background:#0f1219;border:1px solid #232938;border-radius:10px;padding:.75rem;text-align:center}video{width:100%;border-radius:8px;background:#000;margin-bottom:.5rem}.dl{display:block;padding:.5rem;background:#141821;border:1px solid #232938;color:#fff;border-radius:6px;text-decoration:none;font-size:.85rem}#st{text-align:center;margin-top:1rem;color:#a5b4fc}</style></head><body><div class="box"><header><h1>⚡ NoxClips AI</h1><p style="color:#9ca3af">30s Viral Clipper • Auto Face Tracking • Animated Captions</p></header><div class="card"><div class="grid"><div class="b"><span>Duration</span><strong>30 Seconds</strong></div><div class="b"><span>AI Style</span><strong>Mixed Hooks</strong></div><div class="b"><span>Aspect Ratio</span><strong>Auto 9:16</strong></div><div class="b"><span>Captions</span><strong>Animated</strong></div></div><label class="up" for="v"><div id="lbl">📁 Tap to select a video (MP4, MOV)</div><input type="file" id="v" accept="video/*"></label><button class="btn" id="btn" disabled>Generate 30s Clips</button><div id="st"></div></div><div id="res" style="display:none"><a id="zip" class="btn zip" href="#" download>📦 Download All Clips (ZIP)</a><div class="clips" id="cg"></div></div></div><script>const v=document.getElementById('v'),lbl=document.getElementById('lbl'),btn=document.getElementById('btn'),st=document.getElementById('st'),res=document.getElementById('res'),cg=document.getElementById('cg'),zip=document.getElementById('zip');v.onchange=()=>{if(v.files.length){lbl.textContent='Selected: '+v.files.name;btn.disabled=false}};btn.onclick=async()=>{const f=v.files;if(!f)return;btn.disabled=true;res.style.display='none';st.textContent='⏳ Processing video (1-2 mins)...';const fd=new FormData();fd.append('file',f);try{const r=await fetch('/api/process',{method:'POST',body:fd});const d=await r.json();if(!r.ok)throw new Error(d.error||'Failed');cg.innerHTML='';d.clips.forEach((u,i)=>{cg.innerHTML+=`<div class="cc"><video src="${u}" controls></video><a class="dl" href="${u}" download="clip_${i+1}.mp4">⬇️ Download Clip ${i+1}</a></div>`});zip.href=d.zip_url;res.style.display='block';st.textContent='✅ Clips ready!'}catch(e){st.textContent='❌ Error: '+e.message}finally{btn.disabled=false}}</script></body></html>"""

def get_face_x(path: str, start: float, dur: float) -> int:
    cap = cv2.VideoCapture(path)
    fps, w = cap.get(cv2.CAP_PROP_FPS) or 30, int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    fc = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    cap.set(cv2.CAP_PROP_POS_FRAMES, int(start * fps))
    centers = []
    for _ in range(int(dur)):
        ret, frame = cap.read()
        if not ret: break
        faces = fc.detectMultiScale(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), 1.1, 4)
        if len(faces) > 0:
            x, y, fw, fh = max(faces, key=lambda f: f * f)
            centers.append(x + fw // 2)
        cap.set(cv2.CAP_PROP_POS_FRAMES, cap.get(cv2.CAP_PROP_POS_FRAMES) + int(fps))
    cap.release()
    return int(sum(centers) / len(centers)) if centers else w // 2

def make_ass(words: list, offset: float, out: str):
    hdr = "[Script Info]\nScriptType: v4.00+\nPlayResX: 1080\nPlayResY: 1920\n\n[V4+ Styles]\nFormat: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding\nStyle: Default,Arial,85,&H00FFFFFF,&H0000FFFF,&H00000000,&H80000000,-1,0,0,0,100,100,0,0,1,5,2,2,50,50,260,1\n\n[Events]\nFormat: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text\n"
    fmt = lambda s: f"{int(s//3600)}:{int((s%3600)//60):02d}:{s%60:05.2f}"
    ev = []
    for i in range(0, len(words), 3):
        c = words[i:i+3]
        if not c: continue
        s, e = max(0, c["start"] - offset), max(c["start"] - offset + 0.2, c[-1]["end"] - offset)
        txt = " ".join(w["word"].strip().upper() for w in c)
        ev.append(f"Dialogue: 0,{fmt(s)},{fmt(e)},Default,,0,0,0,,{{\\fscx80\\fscy80\\t(0,80,\\fscx105\\fscy105)\\t(80,150,\\fscx100\\fscy100)\\c&H0000FFFF&}}{txt}")
    with open(out, "w", encoding="utf-8") as f: f.write(hdr + "\n".join(ev))

def find_clips(segs: list, dur: int = 30) -> list:
    hooks = {"why", "how", "secret", "never", "stop", "mistake", "truth", "best", "worst", "money", "ai"}
    scored = []
    for s in segs:
        st = s["start"]
        win = [x for x in segs if x["start"] >= st and x["end"] <= st + dur]
        if not win: continue
        txt = " ".join(x["text"].lower() for x in win)
        scored.append((len(txt.split()) + sum(10 for w in hooks if w in txt) + txt.count("?") * 15, st, win))
    scored.sort(key=lambda x: x, reverse=True)
    sel = []
    for _, st, win in scored:
        if all(abs(st - x) >= dur for x in sel): sel.append((st, win))
        if len(sel) == 3: break
    return sel

@app.get("/")
def home():
    return HTMLResponse(UI)

@app.post("/api/process")
async def process(file: UploadFile = File(...)):
    try:
        jid = str(uuid.uuid4())[:8]
        jdir = os.path.join("outputs", jid)
        os.makedirs(jdir, exist_ok=True)
        ipath = os.path.join("uploads", f"{jid}_{file.filename}")
        with open(ipath, "wb") as b: shutil.copyfileobj(file.file, b)
        raw, _ = model.transcribe(ipath, word_timestamps=True)
        segs = [{"start": s.start, "end": s.end, "text": s.text, "words": [{"word": w.word, "start": w.start, "end": w.end} for w in (s.words or [])]} for s in raw]
        clips = find_clips(segs, 30) or [(0.0, segs)]
        cap = cv2.VideoCapture(ipath)
        ow, oh = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        cap.release()
        urls, paths = [], []
        for i, (st, win) in enumerate(clips):
            ass = os.path.join(jdir, f"s_{i}.ass")
            make_ass([w for s in win for w in s.get("words", [])], st, ass)
            fx = get_face_x(ipath, st, 30)
            tw = min(int(oh * (9 / 16)), ow)
            cx = max(0, min(ow - tw, fx - tw // 2))
            vf = f"crop={tw}:{oh}:{cx}:0,scale=1080:1920,ass={ass}"
            out = os.path.join(jdir, f"clip_{i+1}.mp4")
            subprocess.run(["ffmpeg", "-y", "-threads", "1", "-ss", str(st), "-t", "30", "-i", ipath, "-vf", vf, "-c:v", "libx264", "-preset", "ultrafast", "-crf", "26", "-c:a", "aac", "-b:a", "128k", out], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            urls.append(f"/outputs/{jid}/clip_{i+1}.mp4")
            paths.append(out)
        zname = f"clips_{jid}.zip"
        with zipfile.ZipFile(os.path.join(jdir, zname), "w") as z:
            for p in paths:
                if os.path.exists(p): z.write(p, os.path.basename(p))
        if os.path.exists(ipath): os.remove(ipath)
        gc.collect()
        return JSONResponse({"job_id": jid, "clips": urls, "zip_url": f"/outputs/{jid}/{zname}"})
    except Exception as e:
        traceback.print_exc()
        return JSONResponse({"error": str(e)}, status_code=500)
