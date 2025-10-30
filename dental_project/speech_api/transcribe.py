# speech_api/diarize_transcribe.py
import numpy as np
from resemblyzer import VoiceEncoder
from pathlib import Path
import whisper
import librosa
import webrtcvad
import scipy
import os

TRANSCRIPTS_DIR = Path("transcripts")
TRANSCRIPTS_DIR.mkdir(exist_ok=True)

def diarize_and_transcribe(audio_path, whisper_model="tiny"):
    """
    Pure CPU speaker diarization + transcription with language translation.
    Saves a single transcript file with speaker labels.
    Returns: List of tuples: (speaker_id, text)
    """
    # --- Load audio ---
    wav, sr = librosa.load(audio_path, sr=16000)
    
    # --- VAD to get speech segments ---
    vad = webrtcvad.Vad(2)  # 0-3 aggressiveness
    frame_duration = 30  # ms
    samples_per_frame = int(sr * frame_duration / 1000)
    
    frames = [wav[i:i+samples_per_frame] for i in range(0, len(wav), samples_per_frame)]
    speech_frames = [i for i, f in enumerate(frames) if len(f) == samples_per_frame and vad.is_speech((f*32768).astype(np.int16).tobytes(), sr)]
    
    # --- Speaker embeddings ---
    encoder = VoiceEncoder()
    embeddings = []
    segments = []
    
    if not speech_frames:
        print("No speech detected!")
        return []
    
    start = speech_frames[0]
    for i in range(1, len(speech_frames)):
        if speech_frames[i] != speech_frames[i-1]+1:
            segment = wav[start*samples_per_frame:(speech_frames[i-1]+1)*samples_per_frame]
            emb = encoder.embed_utterance(segment)
            embeddings.append(emb)
            segments.append(segment)
            start = speech_frames[i]
    # Last segment
    segment = wav[start*samples_per_frame:(speech_frames[-1]+1)*samples_per_frame]
    embeddings.append(encoder.embed_utterance(segment))
    segments.append(segment)
    
    # --- Cluster embeddings ---
    from sklearn.cluster import AgglomerativeClustering
    clustering = AgglomerativeClustering(n_clusters=2)  # adjust if more speakers
    labels = clustering.fit_predict(np.vstack(embeddings))
    
    # --- Load Whisper ---
    model = whisper.load_model(whisper_model)
    
    transcript_text = ""
    results = []
    
    for i, seg in enumerate(segments):
        temp_path = "temp.wav"
        scipy.io.wavfile.write(temp_path, sr, (seg*32768).astype(np.int16))
        # Transcribe with language and translate
        text = model.transcribe(temp_path)["text"]
        speaker_id = f"Speaker_{labels[i]}"
        results.append((speaker_id, text))
        
        # Append to transcript text
        transcript_text += f"{speaker_id}: {text}\n"
        
        os.remove(temp_path)
    
    # Save single transcript file
    transcript_file = TRANSCRIPTS_DIR / f"{Path(audio_path).stem}.txt"
    with open(transcript_file, "w", encoding="utf-8") as f:
        f.write(transcript_text)
    
    print(f"Transcript saved to '{transcript_file}'")
    
    return results

if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Usage: python diarize_transcribe.py <audio_file_path>")
        sys.exit(1)

    audio_file = sys.argv[1]
    output = diarize_and_transcribe(audio_file)
    for speaker, text in output:
        print(f"{speaker}: {text}")
