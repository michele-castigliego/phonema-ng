import os
import json
import numpy as np
import argparse

# === Argomenti ===
parser = argparse.ArgumentParser()
parser.add_argument("--phonemized_dir", type=str, default="output", help="Directory contenente i file phonemized_*.jsonl")
parser.add_argument("--lang", type=str, required=True, help="Codice lingua (es. it, en)")
parser.add_argument("--phoneme_map_path", type=str, required=True, help="Path al file phoneme_to_index.json")
parser.add_argument("--output_dir", type=str, default="output", help="Dove salvare la matrice aggiornata")
args = parser.parse_args()

# === Caricamento mappa fonemi ===
with open(args.phoneme_map_path, "r") as f:
    phoneme_to_index = json.load(f)
    num_classes = len(phoneme_to_index)

# === Inizializzazione matrice ===
matrix_path = os.path.join(args.output_dir, f"viterbi-matrix_{args.lang}.npy")
if os.path.exists(matrix_path):
    print(f"[i] Matrice esistente trovata: {matrix_path}")
    A = np.load(matrix_path)
else:
    print(f"[i] Creazione nuova matrice: {matrix_path}")
    A = np.zeros((num_classes, num_classes), dtype=np.float32)

counts = np.zeros_like(A)

# === Parsing dei file phonemized_*.jsonl ===
splits = ["train", "dev", "test"]
for split in splits:
    file_path = os.path.join(args.phonemized_dir, f"phonemized_{split}.jsonl")
    if not os.path.exists(file_path):
        print(f"[!] File non trovato: {file_path}")
        continue
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            data = json.loads(line)
            phonemes = data["phonemes"]  # Already a list
            ids = [phoneme_to_index[p] for p in phonemes if p in phoneme_to_index]
            for i in range(1, len(ids)):
                A[ids[i - 1], ids[i]] += 1
                counts[ids[i - 1], ids[i]] += 1

# === Normalizzazione righe ===
for i in range(num_classes):
    total = np.sum(counts[i])
    if total > 0:
        A[i] = A[i] / total
    else:
        A[i, i] = 1.0

# === Salvataggio ===
np.save(matrix_path, A)
print(f"[✓] Matrice Viterbi salvata: {matrix_path}")

