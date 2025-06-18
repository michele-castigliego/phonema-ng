import os
import re
import json
import argparse
import pandas as pd
from num2words import num2words
from phonemizer.libg2p import g2p_with_separator

def normalize_text(text):
    text = text.lower()
    text = re.sub(r'\d+', lambda m: num2words(int(m.group()), lang='it'), text)
    text = re.sub(r"[^\w\sàèéìòù]", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

def phonemize(tsv_path, output_path):
    if not os.path.exists(tsv_path):
        raise FileNotFoundError(f"File non trovato: {tsv_path}")

    df = pd.read_csv(tsv_path, sep="\t")
    clips_dir = os.path.join(os.path.dirname(tsv_path), "clips")

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as fout:
        for i, row in df.iterrows():
            original = row["sentence"]
            normalized = normalize_text(original)

            ipa_string = g2p_with_separator(normalized, sep="|")
            phoneme_list = ipa_string.split("|") if ipa_string else []

            audio_path = os.path.join(clips_dir, row["path"])

            fout.write(json.dumps({
                "sentence": original,
                "normalized": normalized,
                "audio_path": audio_path,
                "phonemes_str": ipa_string,
                "phonemes": phoneme_list
            }, ensure_ascii=False) + "\n")

            if (i + 1) % 100 == 0:
                print(f"Processate {i+1} frasi")

    print(f"\n✅ Fonemizzazione completata. File salvato: {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fonemizza le frasi da un file TSV di Common Voice")
    parser.add_argument("--tsv", required=True, help="Percorso al file .tsv da elaborare")
    parser.add_argument("--out", required=True, help="Percorso al file di output .jsonl")

    args = parser.parse_args()

    phonemize(args.tsv, args.out)

