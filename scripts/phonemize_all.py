import os
import subprocess

def run_phonemizer(tsv_path, output_path):
    print(f"\n📥 Fonemizzazione: {tsv_path}")
    cmd = [
        "python", "scripts/phonemize.py",
        "--tsv", tsv_path,
        "--out", output_path
    ]
    subprocess.run(cmd, check=True)

def main():
    base_dir = "DataSet/cv-corpus-18.0-2024-06-14/it"
    output_dir = "output"

    pairs = [
        ("train.tsv", "phonemized_train.jsonl"),
        ("dev.tsv", "phonemized_dev.jsonl"),
        ("test.tsv", "phonemized_test.jsonl"),
    ]

    for tsv_file, jsonl_file in pairs:
        tsv_path = os.path.join(base_dir, tsv_file)
        out_path = os.path.join(output_dir, jsonl_file)
        run_phonemizer(tsv_path, out_path)

    print("\n✅ Fonemizzazione completata per tutti i file.")

if __name__ == "__main__":
    main()

