import argparse
import json
import librosa
import numpy as np
import os
import soundfile as sf
from pathlib import Path
import pandas as pd
import scipy.signal
import sys
from tqdm import tqdm
from phonemizer.load_config import load_config

def apply_preemphasis(y, coeff=0.97):
    return scipy.signal.lfilter([1, -coeff], [1], y)

def normalize(mel):
    mean = mel.mean()
    std = mel.std()
    return (mel - mean) / (std + 1e-6)

def extract_mel(audio_path, config, preemph=False, normalize_flag=True):
    sr = config["sr"]
    n_fft = config["n_fft"]
    hop_length = config["hop_length"]
    n_mels = config["n_mels"]
    top_db = config.get("top_db", 30)

    y, orig_sr = sf.read(audio_path)
    if y.ndim > 1:
        y = y.mean(axis=1)
    if orig_sr != sr:
        y = librosa.resample(y=y, orig_sr=orig_sr, target_sr=sr)
    y, _ = librosa.effects.trim(y, top_db=top_db)
    if preemph:
        y = apply_preemphasis(y)
    mel = librosa.feature.melspectrogram(y=y, sr=sr, n_fft=n_fft,
                                         hop_length=hop_length, n_mels=n_mels, power=2.0)
    mel_db = librosa.power_to_db(mel, ref=np.max)
    if normalize_flag:
        mel_db = normalize(mel_db)
    return mel_db.T  # (frames, mel)

def main(args):
    config = load_config()
    input_jsonl = Path(args.input_jsonl)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    index_data = []

    with input_jsonl.open("r", encoding="utf-8") as f:
        lines = f.readlines()
        for line in tqdm(lines, desc=f"[{input_jsonl.stem.upper()}]", file=sys.stdout):
            sample = json.loads(line)
            mel = extract_mel(
                audio_path=sample["audio_path"],
                config=config,
                preemph=args.preemphasis,
                normalize_flag=not args.no_norm,
            )
            sample_id = Path(sample["audio_path"]).stem
            out_path = output_dir / f"{sample_id}.npy"
            np.save(out_path, mel)
            index_data.append({
                "id": sample_id,
                "mel_path": str(out_path),
                "n_frames": mel.shape[0],
                "n_mels": mel.shape[1],
                "phonemes": sample["phonemes"],
            })

    df = pd.DataFrame(index_data)
    df.to_csv(args.index_csv, index=False)
    print(f"✅ Salvati {len(df)} spettrogrammi in {output_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_jsonl", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--index_csv", type=str, required=True)
    parser.add_argument("--preemphasis", action="store_true", help="Applica pre-enfasi")
    parser.add_argument("--no_norm", action="store_true", help="Disabilita normalizzazione")
    args = parser.parse_args()
    main(args)

