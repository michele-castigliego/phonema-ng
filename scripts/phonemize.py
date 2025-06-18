import os
import re
import json
import pandas as pd
from espeakng import ESpeakNG
from num2words import num2words

# Config
BASE_PATH = "DataSet/cv-corpus-18.0-2024-06-14/it"
TSV_PATH = os.path.join(BASE_PATH, "train.tsv")
OUTPUT_PATH = "output/phonemized_train.jsonl"
LANGUAGE = "it"
MAX_ROWS = None  # es. 1000 per debug

def normalize_text(text):
    text = text.lower()
    text = re.sub(r'\d+', lambda m: num2words(int(m.group()), lang='it'), text)
    text = re.sub(r"[^\w\sàèéìòù]", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

def main():
    if not os.path.exists(TSV_PATH):
        raise FileNotFoundError(f"File non trovato: {TSV_PATH}")

    df = pd.read_csv(TSV_PATH, sep="\t")
    if MAX_ROWS:
        df = df.head(MAX_ROWS)

    esng = ESpeakNG()
    esng.voice = LANGUAGE

    with open(OUTPUT_PATH, "w", encoding="utf-8") as fout:
        for i, row in df.iterrows():
            original = row["sentence"]
            normalized = normalize_text(original)
            ipa = esng.g2p(normalized, ipa=True)

            fout.write(json.dumps({
                "sentence": original,
                "normalized": normalized,
                "path": row["path"],
                "phonemes": ipa
            }, ensure_ascii=False) + "\n")

            if (i + 1) % 100 == 0:
                print(f"Processate {i+1} frasi")

    print(f"\n✅ Fonemizzazione completata. File salvato: {OUTPUT_PATH}")

if __name__ == "__main__":
    main()

