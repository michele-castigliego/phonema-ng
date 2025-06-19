import os
import argparse
import subprocess

def run_extract(split, args):
    print(f"================= [{split.upper()}] =================")
    jsonl = os.path.join(args.phonemized_dir, f"phonemized_{split}_wav.jsonl")
    output_dir = os.path.join(args.output_dir, split)
    index_csv = os.path.join(args.output_dir, f"{split}_index.csv")

    cmd = [
        "python", "scripts/extract_mels.py",
        "--input_jsonl", jsonl,
        "--output_dir", output_dir,
        "--index_csv", index_csv,
        "--config", args.config,
        "--sr", str(args.sr),
        "--n_fft", str(args.n_fft),
        "--hop_length", str(args.hop_length),
        "--n_mels", str(args.n_mels),
        "--num_workers", str(args.num_workers)
    ]

    if args.no_norm:
        cmd.append("--no_norm")

    subprocess.run(cmd, check=True)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--phonemized_dir", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--config", type=str, default="config.yaml")
    parser.add_argument("--sr", type=int, default=16000)
    parser.add_argument("--n_fft", type=int, default=400)
    parser.add_argument("--hop_length", type=int, default=160)
    parser.add_argument("--n_mels", type=int, default=80)
    parser.add_argument("--no_norm", action="store_true")
    parser.add_argument("--num_workers", type=int, default=1)
    args = parser.parse_args()

    for split in ["train", "dev", "test"]:
        run_extract(split, args)

if __name__ == "__main__":
    main()
