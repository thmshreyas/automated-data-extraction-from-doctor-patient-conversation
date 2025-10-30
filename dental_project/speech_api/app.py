import whisper
from pathlib import Path
import datetime
import sys

# ----------- CONFIG ------------
MODEL_NAME = "medium"      # or "base", "small", "medium", "large"
LANGUAGE = "kn"          # or None for auto-detect
OUTPUT_DIR = Path("transcripts")
OUTPUT_DIR.mkdir(exist_ok=True)
# -------------------------------

def transcribe_audio(audio_path: str):
    """
    Transcribes the given audio file and saves the result in a text file.
    """
    # Load Whisper model
    model = whisper.load_model(MODEL_NAME, device="cuda")

    # Run transcription
    result = model.transcribe(audio_path, task = "translate", language=LANGUAGE)
    text = result["text"].strip()

    # Save to timestamped file
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = OUTPUT_DIR / f"{Path(audio_path).stem}_{timestamp}.txt"
    output_file.write_text(text, encoding="utf-8")

    print(f"✅ Transcription saved to: {output_file}")
    return output_file


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python app.py <audio_file>")
        sys.exit(1)

    audio_file = sys.argv[1]
    transcribe_audio(audio_file)
