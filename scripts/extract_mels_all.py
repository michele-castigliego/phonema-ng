import argparse
import os
import sys
from pathlib import Path
import subprocess

def main(args):
    splits = ["train", "dev", "test"]
    for split in splits:
        jsonl_path = Path(args.phonemized_dir) / f"phonemized_{split}.jsonl"
        out_dir = Path(args.output_dir) / split
        index_csv = Path(args.output_dir) / f"{split}_index.csv"

        cmd = [
            "python", "scripts/extract_mels.py",
            "--input_jsonl", str(jsonl_path),
            "--output_dir", str(out_dir),
            "--index_csv", str(index_csv),
        ]
        if args.preemphasis:
            cmd.append("--preemphasis")
        if args.no_norm:
            cmd.append("--no_norm")

        print(f"▶️ Estrazione {split}...")
        subprocess.run(cmd, check=True, stdout=sys.stdout, stderr=sys.stderr)
        #subprocess.run(cmd, check=True)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--phonemized_dir", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--preemphasis", action="store_true", help="Applica pre-enfasi")
    parser.add_argument("--no_norm", action="store_true", help="Disabilita normalizzazione")
    args = parser.parse_args()
    main(args)

