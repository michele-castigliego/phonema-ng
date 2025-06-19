# Aggiornamento di scripts/phonemize.py per supporto multilingua
import argparse
import json
from pathlib import Path
from phonemizer.libg2p import g2p_with_separator
from phonemizer.normalize import normalize_text
from tqdm import tqdm

def phonemize_line(sentence, lang):
    norm = normalize_text(sentence)
    phonemes_str = g2p_with_separator(norm, sep="|", lang=lang)
    phonemes = phonemes_str.split("|")
    phonemes = [f"<LANG:{lang}>"] + phonemes  # Inserisce token linguistico
    return norm, phonemes_str, phonemes

def main(args):
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with open(args.tsv, "r", encoding="utf-8") as f:
        lines = [line.strip().split("\t") for line in f if line.strip()]
        header = lines[0]
        rows = lines[1:]

    idx_sentence = header.index("sentence")
    idx_path = header.index("path")

    with open(out_path, "w", encoding="utf-8") as fout:
        for row in tqdm(rows, desc="Phonemizing"):
            sentence = row[idx_sentence]
            rel_path = row[idx_path]
            audio_path = str(Path(args.audio_dir) / rel_path)
            normalized, phonemes_str, phonemes = phonemize_line(sentence, args.lang)
            fout.write(json.dumps({
                "sentence": sentence,
                "normalized": normalized,
                "lang": args.lang,
                "audio_path": audio_path,
                "phonemes_str": phonemes_str,
                "phonemes": phonemes,
            }, ensure_ascii=False) + "\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--tsv", type=str, required=True)
    parser.add_argument("--out", type=str, required=True)
    parser.add_argument("--audio_dir", type=str, default="DataSet/cv-corpus-18.0-2024-06-14/it/clips")
    parser.add_argument("--lang", type=str, default="it")
    args = parser.parse_args()
    main(args)

