
import os
import numpy as np
import glob
import argparse
from collections import Counter, defaultdict

# === Argomenti ===
parser = argparse.ArgumentParser()
parser.add_argument("--target_dir", type=str, required=True, help="Directory base dei target frame-level (train/dev/test)")
parser.add_argument("--lang", type=str, required=True, help="Codice lingua, es. 'it', 'en'")
parser.add_argument("--output_dir", type=str, default="output", help="Cartella dove salvare la matrice")
parser.add_argument("--num_classes", type=int, required=True, help="Numero totale di classi fonemiche")
args = parser.parse_args()

# === Path completo della matrice ===
matrix_path = os.path.join(args.output_dir, f"viterbi-matrix_{args.lang}.npy")

# === Caricamento matrice esistente (se presente) ===
if os.path.exists(matrix_path):
    print(f"[i] Trovata matrice esistente: {matrix_path}")
    A = np.load(matrix_path)
else:
    print(f"[i] Nessuna matrice trovata. Creazione nuova.")
    A = np.zeros((args.num_classes, args.num_classes), dtype=np.float32)

counts = np.zeros_like(A)

# === Costruzione da tutti i file .npy ===
transitions = defaultdict(Counter)
for subset in ["train", "dev", "test"]:
    path = os.path.join(args.target_dir, subset)
    for file in glob.glob(os.path.join(path, "*.npy")):
        seq = np.load(file)
        seq = [int(i) for i in seq if 0 <= i < args.num_classes]
        for i in range(1, len(seq)):
            transitions[seq[i - 1]][seq[i]] += 1

# === Aggiunta nuovi conteggi alla matrice A ===
for i in range(args.num_classes):
    total = sum(transitions[i].values())
    if total > 0:
        for j in transitions[i]:
            A[i, j] += transitions[i][j]
            counts[i, j] += transitions[i][j]

# === Ricalcolo probabilità normalizzate ===
for i in range(args.num_classes):
    total = np.sum(counts[i])
    if total > 0:
        A[i] = A[i] / total
    else:
        A[i, i] = 1.0

# === Salvataggio finale ===
np.save(matrix_path, A)
print(f"[✓] Matrice Viterbi aggiornata/salvata: {matrix_path}")

