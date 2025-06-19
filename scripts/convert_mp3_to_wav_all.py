import os
import argparse
import subprocess

def run_conversion(split, args):
    print(f"================= [{split.upper()}] =================")
    jsonl_in = os.path.join(args.phonemized_dir, f"phonemized_{split}.jsonl")
    jsonl_out = os.path.join(args.output_dir, f"phonemized_{split}_wav.jsonl")
    wav_dir = os.path.join(args.output_dir, "wav_trimmed", split)

    cmd = [
        "python", "scripts/convert_mp3_to_wav.py",
        "--input_jsonl", jsonl_in,
        "--output_jsonl", jsonl_out,
        "--output_wav_dir", wav_dir,
        "--config", args.config,
        "--num_workers", str(args.num_workers)
    ]

    if args.preemphasis:
        cmd.append("--preemphasis")

    subprocess.run(cmd, check=True)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--phonemized_dir", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--config", type=str, default="config.yaml")
    parser.add_argument("--num_workers", type=int, default=1)
    parser.add_argument("--preemphasis", action="store_true")
    args = parser.parse_args()

    for split in ["train", "dev", "test"]:
        run_conversion(split, args)

if __name__ == "__main__":
    main()
