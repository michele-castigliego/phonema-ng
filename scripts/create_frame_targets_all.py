import argparse
import subprocess

def run_create(split, args):
    print(f"\n================= [{split.upper()}] =================")
    cmd = [
        "python", "scripts/create_frame_targets.py",
        "--input_jsonl", f"{args.phonemized_dir}/phonemized_{split}.jsonl",
        "--mel_dir", f"{args.mels_dir}/{split}",
        "--output_dir", f"{args.output_dir}/{split}"
    ]
    subprocess.run(cmd, check=True)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--phonemized_dir", required=True)
    parser.add_argument("--mels_dir", required=True)
    parser.add_argument("--output_dir", required=True)
    args = parser.parse_args()

    for split in ["train", "dev", "test"]:
        run_create(split, args)

if __name__ == "__main__":
    main()

