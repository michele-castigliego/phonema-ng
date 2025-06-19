import argparse
import os
import json
from pathlib import Path
import numpy as np
import librosa
import soundfile as sf
import yaml
from tqdm import tqdm
from joblib import Parallel, delayed

def load_config(path):
    with open(path, "r") as f:
        return yaml.safe_load(f)

def process_and_store(entry, args, config, out_dir):
    audio_path = entry["audio_path"]
    phonemes = entry["phonemes"]
    uid = Path(audio_path).stem
    out_path = out_dir / f"{uid}.npz"

    try:
        y, sr = sf.read(audio_path)
        if len(y.shape) > 1:
            y = y.mean(axis=1)

        if sr != args.sr:
            return None

        S = librosa.feature.melspectrogram(
            y=y,
            sr=sr,
            n_fft=args.n_fft,
            hop_length=args.hop_length,
            n_mels=args.n_mels,
            power=2.0,
        )

        if not args.no_norm:
            S = librosa.power_to_db(S, ref=np.max)
            S = (S - S.mean()) / (S.std() + 1e-6)

        mel = S.T.astype(np.float32)

        np.savez_compressed(out_path, mel=mel, phonemes=phonemes, audio_path=audio_path)

        return {
            "id": uid,
            "audio_path": audio_path,
            "phonemes": phonemes,
            "num_frames": mel.shape[0]
        }
    except Exception as e:
        print(f"[Errore] {audio_path}: {e}")
        return None

def main(args):
    config = load_config(args.config)

    with open(args.input_jsonl, "r", encoding="utf-8") as f:
        entries = [json.loads(line.strip()) for line in f if line.strip()]

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    results = Parallel(n_jobs=args.num_workers)(
        delayed(process_and_store)(entry, args, config, out_dir) for entry in tqdm(entries, desc="Processing")
    )

    results = [r for r in results if r]

    if args.index_csv:
        with open(args.index_csv, "w", encoding="utf-8") as f:
            f.write("id,audio_path,phonemes,num_frames\n")
            for entry in results:
                phonemes_str = json.dumps(entry["phonemes"], ensure_ascii=False)
                f.write(f"{entry['id']},{entry['audio_path']},{phonemes_str},{entry['num_frames']}\n")

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
