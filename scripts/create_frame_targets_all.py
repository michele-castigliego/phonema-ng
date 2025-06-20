import argparse
import subprocess
import yaml
import os

def load_config(path):
    with open(path, "r") as f:
        return yaml.safe_load(f)

def run_create(split, cfg):
    print(f"\n================= [{split.upper()}] =================")
    cmd = [
        "python", "scripts/create_frame_targets.py",
        "--input_jsonl", os.path.join(cfg["phonemized_dir"], f"phonemized_{split}.jsonl"),
        "--mel_dir", os.path.join(cfg["mel_output_dir"], split),
        "--output_dir", os.path.join(cfg["frame_targets_dir"], split)
    ]
    subprocess.run(cmd, check=True)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="config.yaml")
    args = parser.parse_args()

    cfg = load_config(args.config)
    for split in ["train", "dev", "test"]:
        run_create(split, cfg)

if __name__ == "__main__":
    main()

