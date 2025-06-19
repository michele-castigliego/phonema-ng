import os
import argparse
import subprocess
import yaml

def load_config(config_path):
    with open(config_path, "r") as f:
        return yaml.safe_load(f)

def run_extract(split, cfg, args):
    print(f"================= [{split.upper()}] =================")
    jsonl = os.path.join(cfg["wav_output_dir"], f"phonemized_{split}_wav.jsonl")
    output_dir = os.path.join(cfg["mel_output_dir"], split)
    index_csv = os.path.join(cfg["mel_output_dir"], f"{split}_index.csv")

    cmd = [
        "python", "scripts/extract_mels.py",
        "--input_jsonl", jsonl,
        "--output_dir", output_dir,
        "--index_csv", index_csv,
        "--config", args.config,
        "--sr", str(cfg["sr"]),
        "--n_fft", str(cfg["n_fft"]),
        "--hop_length", str(cfg["hop_length"]),
        "--n_mels", str(cfg["n_mels"]),
        "--num_workers", str(args.num_workers)
    ]

    if args.no_norm:
        cmd.append("--no_norm")

    subprocess.run(cmd, check=True)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="config.yaml")
    parser.add_argument("--no_norm", action="store_true")
    parser.add_argument("--num_workers", type=int, default=1)
    args = parser.parse_args()

    cfg = load_config(args.config)
    for split in ["train", "dev", "test"]:
        run_extract(split, cfg, args)

if __name__ == "__main__":
    main()

