import os
import json
import numpy as np
import pandas as pd
from tqdm import tqdm
import librosa
from phonemizer.load_config import load_config

# Caricamento configurazione
config = load_config()

# Parametri da config
dataset_dir = config["dataset_dir"]
phonemized_dir = config["phonemized_dir"]
output_dir = config["mel_output_dir"]

sr = config["sr"]
n_fft = config["n_fft"]
hop_length = config["hop_length"]
n_mels = config["n_mels"]
top_db = config.get("top_db", 30)

os.environ["LIBROSA_AUDIO_BACKEND"] = "ffmpeg"

splits = ["train", "dev", "test"]

def extract_mel_spectrogram(audio_path):
    y_raw, _ = librosa.load(audio_path, sr=sr)
    duration_raw = librosa.get_duration(y=y_raw, sr=sr)

    if len(y_raw) == 0:
        return None, 0.0, 0.0, 0, False

    y_trimmed, _ = librosa.effects.trim(y_raw, top_db=top_db)
    duration_trimmed = librosa.get_duration(y=y_trimmed, sr=sr)

    if len(y_trimmed) == 0:
        return None, duration_raw, 0.0, len(y_raw), True

    mel = librosa.feature.melspectrogram(y=y_trimmed, sr=sr, n_fft=n_fft,
                                          hop_length=hop_length, n_mels=n_mels)
    mel_db = librosa.power_to_db(mel, ref=np.max)

    return mel_db.T, duration_raw, duration_trimmed, len(y_raw) - len(y_trimmed), True

def process_split(split):
    input_jsonl = os.path.join(phonemized_dir, f"phonemized_{split}.jsonl")
    split_output_dir = os.path.join(output_dir, split)
    index_csv = os.path.join(output_dir, f"{split}_index.csv")

    os.makedirs(split_output_dir, exist_ok=True)

    with open(input_jsonl, "r", encoding="utf-8") as f:
        data = [json.loads(line) for line in f]

    records = []

    for entry in tqdm(data, desc=f"[{split.upper()}]"):
        mel, dur_raw, dur_trim, samples_removed, trimmed = extract_mel_spectrogram(entry['audio_path'])

        if mel is None:
            print(f"⚠️  File vuoto o danneggiato: {entry['audio_path']}")
            continue

        audio_id = os.path.splitext(os.path.basename(entry['audio_path']))[0]
        mel_filename = f"{audio_id}.npz"
        mel_path = os.path.join(split_output_dir, mel_filename)

        np.savez_compressed(mel_path, mel=mel, phonemes=entry['phonemes'])

        records.append({
            "id": audio_id,
            "audio_path": entry['audio_path'],
            "mel_path": mel_path,
            "n_frames": mel.shape[0],
            "n_phonemes": len(entry['phonemes']),
            "duration_sec_raw": round(dur_raw, 3),
            "duration_sec_trimmed": round(dur_trim, 3),
            "samples_removed": samples_removed,
            "trimmed": trimmed
        })

    df = pd.DataFrame(records)
    df.to_csv(index_csv, index=False)

    print(f"✅ {split} completato: {len(records)} file")
    print(f"📄 Index CSV: {index_csv}")

def main():
    for split in splits:
        process_split(split)

if __name__ == "__main__":
    main()

