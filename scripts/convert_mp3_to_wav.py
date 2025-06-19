import argparse
import json
import os
from pathlib import Path

import librosa
import soundfile as sf
import yaml
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor

def load_config(path):
    with open(path, "r") as f:
        return yaml.safe_load(f)

def preemphasis(signal, coef=0.97):
    return librosa.effects.preemphasis(signal, coef=coef)

def process_entry(entry, out_wav_dir, args, config):
    in_path = entry["audio_path"]
    uid = Path(in_path).stem
    out_path = out_wav_dir / f"{uid}.wav"

    try:
        y, sr_orig = librosa.load(in_path, sr=None, mono=True)
        y, _ = librosa.effects.trim(y, top_db=config.get("top_db", 30))
        if sr_orig != config["sr"]:
            y = librosa.resample(y, orig_sr=sr_orig, target_sr=config["sr"])
        if not args.no_preemphasis:
            y = preemphasis(y)
        sf.write(out_path, y, config["sr"])
    except Exception as e:
        print(f"[Errore] {in_path}: {e}")
        return None

    new_entry = entry.copy()
    new_entry["audio_path"] = str(out_path)
    return new_entry

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_jsonl", type=str, required=True)
    parser.add_argument("--output_jsonl", type=str, required=True)
    parser.add_argument("--output_wav_dir", type=str, required=True)
    parser.add_argument("--config", type=str, default="config.yaml")
    parser.add_argument("--num_workers", type=int, default=1)
    parser.add_argument("--no_preemphasis", action="store_true", help="Disattiva la pre-enfasi (attiva di default)")
    args = parser.parse_args()

    config = load_config(args.config)
    out_wav_dir = Path(args.output_wav_dir)
    out_wav_dir.mkdir(parents=True, exist_ok=True)

    with open(args.input_jsonl, "r", encoding="utf-8") as f:
        entries = [json.loads(line.strip()) for line in f if line.strip()]

    new_entries = []

    with ThreadPoolExecutor(max_workers=args.num_workers) as executor:
        futures = [executor.submit(process_entry, entry, out_wav_dir, args, config) for entry in entries]
        for fut in tqdm(futures, total=len(futures), desc="Converting"):
            result = fut.result()
            if result:
                new_entries.append(result)

    with open(args.output_jsonl, "w", encoding="utf-8") as f:
        for e in new_entries:
            f.write(json.dumps(e, ensure_ascii=False) + "\n")

if __name__ == "__main__":
    main()

