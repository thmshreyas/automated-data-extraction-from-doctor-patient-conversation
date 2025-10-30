from fastapi import FastAPI, UploadFile, File
import whisper
import tempfile
from pathlib import Path
import datetime

app = FastAPI()

# Default model (can be "tiny", "base", "small", "medium", "large")
MODEL_NAME = "medium"
model = whisper.load_model(MODEL_NAME)

# Folder to save transcriptions
OUTPUT_DIR = Path("transcriptions")
OUTPUT_DIR.mkdir(exist_ok=True)

@app.post("/transcribe")
async def transcribe(file: UploadFile = File(...), language: str = "kn"):
    # Save uploaded file to a temp location
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp:
        tmp.write(await file.read())
        tmp_path = tmp.name

    # Run transcription
    result = model.transcribe(tmp_path, language=language)
    text = result["text"]

    # Save text to a file
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    text_file = OUTPUT_DIR / f"{file.filename}_{timestamp}.txt"
    text_file.write_text(text, encoding="utf-8")

    return {
        "text": text,
        "saved_file": str(text_file)
    }
