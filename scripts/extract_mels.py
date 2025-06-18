import os
import json
import argparse
import numpy as np
import pandas as pd
import librosa
from tqdm import tqdm
from phonemizer.load_config import load_config

# Caricamento configurazione
config = load_config()

sr = config["sr"]
n_fft = config["n_fft"]
hop_length = config["hop_length"]
n_mels = config["n_mels"]
top_db = config.get("top_db", 30)

os.environ["LIBROSA_AUDIO_BACKEND"] = "ffmpeg"

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

def main(args):
    with open(args.input_jsonl, 'r', encoding='utf-8') as f:
        data = [json.loads(line) for line in f]

    os.makedirs(args.output_dir, exist_ok=True)
    records = []

    for entry in tqdm(data, desc="Estrazione mel-spectrogrammi"):
        mel, dur_raw, dur_trim, samples_removed, trimmed = extract_mel_spectrogram(entry['audio_path'])

        if mel is None:
            print(f"⚠️  File vuoto o danneggiato: {entry['audio_path']}")
            continue

        audio_id = os.path.splitext(os.path.basename(entry['audio_path']))[0]
        mel_filename = f"{audio_id}.npz"
        mel_path = os.path.join(args.output_dir, mel_filename)

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
    df.to_csv(args.index_csv, index=False)

    print(f"\n✅ Estrazione completata: {len(records)} file validi")
    print(f"📁 Directory spettrogrammi: {args.output_dir}")
    print(f"📄 Index CSV: {args.index_csv}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Estrai mel-spectrogrammi e salva ogni frase separatamente")
    parser.add_argument("--input_jsonl", required=True, help="File JSONL con path audio e fonemi")
    parser.add_argument("--output_dir", required=True, help="Directory per salvare i .npz")
    parser.add_argument("--index_csv", required=True, help="Output CSV con metadati")
    args = parser.parse_args()
    main(args)

