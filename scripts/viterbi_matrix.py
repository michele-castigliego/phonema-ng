import os
import argparse

from utils.viterbi import build_viterbi_matrix

# === Argomenti ===
parser = argparse.ArgumentParser()
parser.add_argument("--phonemized_dir", type=str, default="output", help="Directory contenente i file phonemized_*.jsonl")
parser.add_argument("--lang", type=str, required=True, help="Codice lingua (es. it, en)")
parser.add_argument("--phoneme_map_path", type=str, required=True, help="Path al file phoneme_to_index.json")
parser.add_argument("--output_dir", type=str, default="output", help="Dove salvare la matrice aggiornata")
args = parser.parse_args()

build_viterbi_matrix(
    phonemized_dir=args.phonemized_dir,
    phoneme_map_path=args.phoneme_map_path,
    lang=args.lang,
    output_dir=args.output_dir,
)

matrix_path = os.path.join(args.output_dir, f"viterbi-matrix_{args.lang}.npy")
print(f"[✓] Matrice Viterbi salvata: {matrix_path}")

