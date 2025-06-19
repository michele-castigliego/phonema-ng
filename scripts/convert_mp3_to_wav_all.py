import os
import argparse
import subprocess
import yaml

def load_config(path):
    with open(path, "r") as f:
        return yaml.safe_load(f)

def run_conversion(split, cfg, args):
    print(f"================= [{split.upper()}] =================")
    jsonl_in = os.path.join(cfg["phonemized_dir"], f"phonemized_{split}.jsonl")
    jsonl_out = os.path.join(cfg["wav_output_dir"], f"phonemized_{split}_wav.jsonl")
    wav_dir = os.path.join(cfg["wav_output_dir"], split)

    cmd = [
        "python", "scripts/convert_mp3_to_wav.py",
        "--input_jsonl", jsonl_in,
        "--output_jsonl", jsonl_out,
        "--output_wav_dir", wav_dir,
        "--config", args.config,
        "--num_workers", str(args.num_workers)
    ]

    if args.no_preemphasis:
        cmd.append("--no_preemphasis")

    subprocess.run(cmd, check=True)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="config.yaml")
    parser.add_argument("--num_workers", type=int, default=1)
    parser.add_argument("--no_preemphasis", action="store_true")
    args = parser.parse_args()

    cfg = load_config(args.config)
    for split in ["train", "dev", "test"]:
        run_conversion(split, cfg, args)

if __name__ == "__main__":
    main()

