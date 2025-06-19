import argparse
import os
import json
from pathlib import Path
import numpy as np
import librosa
import soundfile as sf
import yaml
from tqdm import tqdm
import pandas as pd

def load_config(path):
    with open(path, "r") as f:
        return yaml.safe_load(f)

def preemphasis(signal, coef=0.97):
    return np.append(signal[0], signal[1:] - coef * signal[:-1])

def process_sample(entry, args, config):
    audio_path = entry["audio_path"]
    phonemes = entry["phonemes"]
    uid = Path(audio_path).stem

    try:
        y, sr_orig = sf.read(audio_path)
        if len(y.shape) > 1:
            y = y.mean(axis=1)
    except Exception as e:
        print(f"[Errore] Impossibile leggere {audio_path}: {e}")
        return None

    duration_raw = len(y) / sr_orig

    if sr_orig != args.sr:
        y = librosa.resample(y, orig_sr=sr_orig, target_sr=args.sr)

    n_samples_before = len(y)
    y_trimmed, index = librosa.effects.trim(y, top_db=config.get("top_db", 30))
    n_samples_after = len(y_trimmed)

    trimmed = n_samples_before != n_samples_after
    samples_removed = n_samples_before - n_samples_after
    duration_trimmed = n_samples_after / args.sr

    if args.preemphasis:
        y_trimmed = preemphasis(y_trimmed)

    S = librosa.feature.melspectrogram(
        y=y_trimmed,
        sr=args.sr,
        n_fft=args.n_fft,
        hop_length=args.hop_length,
        n_mels=args.n_mels,
        power=2.0,
    )

    if not args.no_norm:
        S = librosa.power_to_db(S, ref=np.max)
        S = (S - S.mean()) / (S.std() + 1e-6)

    mel = S.T

    return {
        "mel": mel.astype(np.float32),
        "phonemes": phonemes,
        "audio_path": audio_path,
        "id": uid,
        "n_frames": mel.shape[0],
        "n_phonemes": len(phonemes),
        "duration_sec_raw": round(duration_raw, 3),
        "duration_sec_trimmed": round(duration_trimmed, 3),
        "samples_removed": samples_removed,
        "trimmed": trimmed
    }

def main(args):
    config = load_config(args.config)

    with open(args.input_jsonl, "r", encoding="utf-8") as f:
        entries = [json.loads(line.strip()) for line in f if line.strip()]

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    records = []

    for entry in tqdm(entries, desc="Processing samples"):
        result = process_sample(entry, args, config)
        if result is None:
            continue

        out_path = out_dir / f"{result['id']}.npz"
        np.savez_compressed(out_path, mel=result["mel"], phonemes=result["phonemes"], audio_path=result["audio_path"])

        records.append({
            "id": result["id"],
            "audio_path": result["audio_path"],
            "mel_path": str(out_path),
            "n_frames": result["n_frames"],
            "n_phonemes": result["n_phonemes"],
            "duration_sec_raw": result["duration_sec_raw"],
            "duration_sec_trimmed": result["duration_sec_trimmed"],
            "samples_removed": result["samples_removed"],
            "trimmed": result["trimmed"]
        })

    if args.index_csv:
        df = pd.DataFrame(records)
        df.to_csv(args.index_csv, index=False)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_jsonl", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--index_csv", type=str, default=None)
    parser.add_argument("--config", type=str, default="config.yaml")
    parser.add_argument("--sr", type=int, default=16000)
    parser.add_argument("--n_fft", type=int, default=400)
    parser.add_argument("--hop_length", type=int, default=160)
    parser.add_argument("--n_mels", type=int, default=80)
    parser.add_argument("--preemphasis", action="store_true")
    parser.add_argument("--no_norm", action="store_true")
    args = parser.parse_args()
    main(args)

