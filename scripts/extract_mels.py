import os
import json
import argparse
import numpy as np
import pandas as pd
import librosa
from tqdm import tqdm

os.environ["LIBROSA_AUDIO_BACKEND"] = "ffmpeg"

def extract_mel_spectrogram(audio_path, sr, n_fft, hop_length, n_mels):
    y, _ = librosa.load(audio_path, sr=sr)
    if len(y) == 0:
        return None, 0.0
    mel = librosa.feature.melspectrogram(y=y, sr=sr, n_fft=n_fft,
                                          hop_length=hop_length, n_mels=n_mels)
    mel_db = librosa.power_to_db(mel, ref=np.max)
    duration = librosa.get_duration(y=y, sr=sr)
    return mel_db.T, duration  # (frames, bins), durata

def main(args):
    with open(args.input_jsonl, 'r', encoding='utf-8') as f:
        data = [json.loads(line) for line in f]

    os.makedirs(args.output_dir, exist_ok=True)
    records = []

    for entry in tqdm(data, desc="Estrazione mel-spectrogrammi"):
        mel, duration = extract_mel_spectrogram(
            audio_path=entry['audio_path'],
            sr=args.sr,
            n_fft=args.n_fft,
            hop_length=args.hop_length,
            n_mels=args.n_mels
        )

        if mel is None:
            print(f"⚠️  File vuoto o danneggiato: {entry['audio_path']}")
            continue

        # ID univoco (basename senza estensione)
        audio_id = os.path.splitext(os.path.basename(entry['audio_path']))[0]
        mel_filename = f"{audio_id}.npz"
        mel_path = os.path.join(args.output_dir, mel_filename)

        np.savez_compressed(mel_path,
                            mel=mel,
                            phonemes=entry['phonemes'])

        records.append({
            "id": audio_id,
            "audio_path": entry['audio_path'],
            "mel_path": mel_path,
            "duration_sec": round(duration, 3),
            "n_frames": mel.shape[0],
            "n_phonemes": len(entry['phonemes']),
        })

    # Salva index CSV
    df = pd.DataFrame(records)
    df.to_csv(args.index_csv, index=False)

    print(f"\n✅ Estrazione completata.")
    print(f"🧱 File salvati: {len(records)}")
    print(f"📁 Directory spettrogrammi: {args.output_dir}")
    print(f"📄 Index CSV: {args.index_csv}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Estrai mel-spectrogrammi e salvali singolarmente")
    parser.add_argument("--input_jsonl", required=True, help="Path al file .jsonl generato da phonemizer")
    parser.add_argument("--output_dir", required=True, help="Cartella dove salvare i file .npz individuali")
    parser.add_argument("--index_csv", required=True, help="File CSV con metadati (path, frame, fonemi, durata)")
    parser.add_argument("--sr", type=int, default=16000, help="Sampling rate")
    parser.add_argument("--n_fft", type=int, default=400, help="Dimensione FFT")
    parser.add_argument("--hop_length", type=int, default=160, help="Hop tra finestre")
    parser.add_argument("--n_mels", type=int, default=80, help="Numero di mel-bin")

    args = parser.parse_args()
    main(args)

