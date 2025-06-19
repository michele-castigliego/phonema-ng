import argparse
import os
import json
from pathlib import Path
import numpy as np
import librosa
import soundfile as sf
import yaml
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor

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
        y, sr = sf.read(audio_path)
        if len(y.shape) > 1:
            y = y.mean(axis=1)
    except Exception as e:
        print(f"[Errore] Impossibile leggere {audio_path}: {e}")
        return None

    if sr != args.sr:
        print(f"[Warning] Sampling rate mismatch in {audio_path}, expected {args.sr}, got {sr}")
        return None

    S = librosa.feature.melspectrogram(
        y=y,
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
        "id": uid
    }

def process_and_store(entry, args, config, out_dir):
    result = process_sample(entry, args, config)
    if result is None:
        return None

    out_path = out_dir / f"{result['id']}.npz"
    np.savez_compressed(out_path, mel=result["mel"], phonemes=result["phonemes"], audio_path=result["audio_path"])
    return result["id"]

def main(args):
    config = load_config(args.config)

    with open(args.input_jsonl, "r", encoding="utf-8") as f:
        entries = [json.loads(line.strip()) for line in f if line.strip()]

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    index_lines = []

    with ThreadPoolExecutor(max_workers=args.num_workers) as executor:
        futures = [executor.submit(process_and_store, e, args, config, out_dir) for e in entries]
        for future in tqdm(futures, total=len(futures), desc="Processing samples"):
            res = future.result()
            if res:
                index_lines.append(res)

    if args.index_csv:
        with open(args.index_csv, "w") as f:
            f.write("id\n" + "\n".join(index_lines))

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
    parser.add_argument("--no_norm", action="store_true")
    parser.add_argument("--num_workers", type=int, default=1)
    args = parser.parse_args()
    main(args)
