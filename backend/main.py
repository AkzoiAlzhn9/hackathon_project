import os, uuid, shutil, re, random
from fastapi import FastAPI, UploadFile, File, BackgroundTasks, Body
from fastapi.middleware.cors import CORSMiddleware
from moviepy.video.io.VideoFileClip import VideoFileClip
from faster_whisper import WhisperModel
from rake_nltk import Rake
import nltk

# Авто-загрузка необходимых ресурсов NLTK
try:
    nltk.download('stopwords')
    nltk.download('punkt')
    nltk.download('punkt_tab') # Исправляет твою ошибку
except:
    pass

app = FastAPI()
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

STORAGE = "storage"
os.makedirs(STORAGE, exist_ok=True)
db = {}

# Грузим модельку (tiny — самая быстраy)
whisper_model = WhisperModel("tiny", device="cpu", compute_type="int8")

def generate_quiz(text):
    """Реальное извлечение смыслов для квиза"""
    rake = Rake(language="russian")
    rake.extract_keywords_from_text(text)
    keywords = rake.get_ranked_phrases()
    
    # Если нейронка нашла мало слов, берем простые варианты
    if not keywords or len(keywords[0]) < 3:
        ans = "Суть урока"
        opts = ["Суть урока", "Введение", "Практика"]
    else:
        ans = keywords[0].capitalize()
        distractors = ["Теория", "Общий обзор", "Пример", "Методика"]
        opts = [ans] + random.sample(distractors, 2)
    
    random.shuffle(opts)
    return {"q": "Что было главным в этом фрагменте?", "opts": opts, "a": ans}

def process_video_logic(vid, path):
    """Мясо обработки: текст -> смысл -> квизы"""
    try:
        # Распознаем речь
        segments_raw, _ = whisper_model.transcribe(path, language="ru")
        full_list = list(segments_raw)
        
        segs = []
        temp_text = ""
        start_t = 0
        
        for i, item in enumerate(full_list):
            temp_text += " " + item.text
            # Режем блоки по 45 секунд
            if (item.end - start_t) > 45 or i == len(full_list) - 1:
                quiz_data = generate_quiz(temp_text)
                segs.append({
                    "id": str(uuid.uuid4()),
                    "start_time": start_t,
                    "end_time": item.end,
                    "topic": f"Глава {len(segs)+1}: {quiz_data['a'][:15]}...",
                    "quiz": quiz_data,
                    "passed": False
                })
                start_t = item.end
                temp_text = ""
        
        db[vid] = {"status": "completed", "segments": segs}
        print(f"--- ОБРАБОТКА ЗАВЕРШЕНА ДЛЯ {vid} ---")
    except Exception as e:
        db[vid] = {"status": "failed", "error": str(e)}

@app.post("/video/upload")
async def upload(bg: BackgroundTasks, file: UploadFile = File(...)):
    vid = str(uuid.uuid4())
    path = os.path.join(STORAGE, f"{vid}.mp4")
    with open(path, "wb") as f:
        shutil.copyfileobj(file.file, f)
    db[vid] = {"status": "processing"}
    bg.add_task(process_video_logic, vid, path)
    return {"video_id": vid}

@app.get("/video/{vid}/segments")
async def get_segs(vid: str):
    return db.get(vid, {"status": "not_found"})

@app.post("/segment/{sid}/answer")
async def check(vid: str, sid: str, data: dict = Body(...)):
    if vid in db:
        for s in db[vid]["segments"]:
            if s["id"] == sid:
                res = s["quiz"]["a"].lower() == data.get("answer", "").lower()
                if res: s["passed"] = True
                return {"correct": res}
    return {"correct": False}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)