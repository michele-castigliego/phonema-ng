import os
import argparse
import subprocess

def run_phonemize(split, args):
    print(f"================= [{split.upper()}] =================")
    tsv = os.path.join(args.dataset_dir, f"{split}.tsv")
    out_path = os.path.join(args.output_dir, f"phonemized_{split}.jsonl")

    cmd = [
        "python", "scripts/phonemize.py",
        "--tsv", tsv,
        "--out", out_path,
        "--audio_dir", os.path.join(args.dataset_dir, "clips"),
        "--lang", args.lang
    ]

    subprocess.run(cmd, check=True)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_dir", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--lang", type=str, default="it")
    args = parser.parse_args()

    for split in ["train", "dev", "test"]:
        run_phonemize(split, args)

if __name__ == "__main__":
    main()

