# scripts/create_shuffled_index_all.py

import os
import json
import random
import argparse
import yaml

def generate_shuffled_index(jsonl_path, output_path, seed=42):
    with open(jsonl_path, "r", encoding="utf-8") as f:
        lines = f.readlines()

    audio_ids = []
    for line in lines:
        sample = json.loads(line)
        audio_path = sample.get("audio_path")
        if audio_path:
            audio_id = os.path.splitext(os.path.basename(audio_path))[0]
            audio_ids.append(audio_id)

    random.seed(seed)
    random.shuffle(audio_ids)

    with open(output_path, "w", encoding="utf-8") as out_f:
        out_f.write("audio_id\n")
        for audio_id in audio_ids:
            out_f.write(f"{audio_id}\n")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="config.yaml")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)

    phonemized_dir = cfg["phonemized_dir"]

    for split in ["train", "dev", "test"]:
        jsonl_path = os.path.join(phonemized_dir, f"phonemized_{split}.jsonl")
        output_path = os.path.join(phonemized_dir, f"{split}_index.csv")
        print(f"➡️  Scrivo {output_path}...")
        generate_shuffled_index(jsonl_path, output_path, seed=args.seed)

if __name__ == "__main__":
    main()

