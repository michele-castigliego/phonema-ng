import os
import subprocess
import argparse

# Script da richiamare per singola estrazione
EXTRACT_SCRIPT = "scripts/extract_mels.py"

def process_split(split, tsv_path, phonemized_jsonl, out_dir, out_index, args):
    print(f"\n🔹 Estrazione per split: {split}")

    if not os.path.exists(phonemized_jsonl):
        print(f"⚠️  Fonemizzazione mancante per {split}, per favore esegui prima phonemize.py")
        return

    command = [
        "python", EXTRACT_SCRIPT,
        "--input_jsonl", phonemized_jsonl,
        "--output_dir", os.path.join(out_dir, split),
        "--index_csv", out_index,
        "--sr", str(args.sr),
        "--n_fft", str(args.n_fft),
        "--hop_length", str(args.hop_length),
        "--n_mels", str(args.n_mels)
    ]

    subprocess.run(command, check=True)

def main(args):
    splits = ["train", "dev", "test"]

    for split in splits:
        tsv_path = os.path.join(args.dataset_dir, f"{split}.tsv")
        jsonl_path = os.path.join(args.phonemized_dir, f"phonemized_{split}.jsonl")
        out_dir_split = os.path.join(args.output_dir, split)
        index_csv = os.path.join(args.output_dir, f"{split}_index.csv")

        process_split(split, tsv_path, jsonl_path, args.output_dir, index_csv, args)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Estrai mel-spectrogrammi per train/dev/test")
    parser.add_argument("--dataset_dir", required=True, help="Directory contenente i file TSV di Common Voice")
    parser.add_argument("--phonemized_dir", required=True, help="Cartella contenente i file .jsonl fonemizzati")
    parser.add_argument("--output_dir", required=True, help="Cartella dove salvare gli .npz e gli index.csv")
    parser.add_argument("--sr", type=int, default=16000)
    parser.add_argument("--n_fft", type=int, default=400)
    parser.add_argument("--hop_length", type=int, default=160)
    parser.add_argument("--n_mels", type=int, default=80)

    args = parser.parse_args()
    main(args)

