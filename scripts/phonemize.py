import argparse
import json
import os
import csv
from tqdm import tqdm

from phonemizer.normalize import normalize_text
from phonemizer.libg2p import g2p_with_separator, check_language_supported

parser = argparse.ArgumentParser()
parser.add_argument("--tsv", type=str, required=True)
parser.add_argument("--out", type=str, required=True)
parser.add_argument("--audio_dir", type=str, required=True)
parser.add_argument("--lang", type=str, default="it")
args = parser.parse_args()

if not check_language_supported(args.lang):
    print(f"Errore: la lingua '{args.lang}' non è supportata da espeak-ng.")
    exit(1)

with open(args.tsv, "r", encoding="utf-8") as f_in:
    reader = list(csv.DictReader(f_in, delimiter="\t"))

with open(args.out, "w", encoding="utf-8") as f_out:
    for row in tqdm(reader, desc="Phonemizing", unit="sent"):
        sentence = row["sentence"]
        path = row["path"]
        normalized = normalize_text(sentence)
        phonemes_str = g2p_with_separator(normalized, sep="|", lang=args.lang)
        phonemes = [f"<LANG:{args.lang}>"] + phonemes_str.split("|")

        json_line = {
            "sentence": sentence,
            "normalized": normalized,
            "lang": args.lang,
            "audio_path": os.path.join(args.audio_dir, path),
            "phonemes_str": phonemes_str,
            "phonemes": phonemes
        }
        f_out.write(json.dumps(json_line, ensure_ascii=False) + "\n")

