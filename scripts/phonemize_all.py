import os
import argparse
import subprocess
import yaml

def load_config(config_path):
    with open(config_path, "r") as f:
        return yaml.safe_load(f)

def run_phonemize(split, cfg, lang):
    print(f"================= [{split.upper()}] =================")
    tsv = os.path.join(cfg["dataset_dir"], f"{split}.tsv")
    out_path = os.path.join(cfg["phonemized_dir"], f"phonemized_{split}.jsonl")

    cmd = [
        "python", "scripts/phonemize.py",
        "--tsv", tsv,
        "--out", out_path,
        "--audio_dir", os.path.join(cfg["dataset_dir"], "clips"),
        "--lang", lang
    ]
    subprocess.run(cmd, check=True)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="config.yaml")
    parser.add_argument("--lang", type=str, default="it")
    args = parser.parse_args()

    cfg = load_config(args.config)
    for split in ["train", "dev", "test"]:
        run_phonemize(split, cfg, args.lang)

if __name__ == "__main__":
    main()

