import os
import argparse
import numpy as np
from tqdm import tqdm

def verify_npz_files(mel_dir):
    print(f"\n🔍 Verifica mel-spectrogrammi in: {mel_dir}")
    for fname in tqdm(os.listdir(mel_dir)):
        if not fname.endswith(".npz"):
            continue
        fpath = os.path.join(mel_dir, fname)
        try:
            data = np.load(fpath)
            _ = data["mel"]
        except Exception as e:
            print(f"[❌ ERRORE] {fpath} → {e}")

def verify_npy_files(target_dir):
    print(f"\n🔍 Verifica target in: {target_dir}")
    for fname in tqdm(os.listdir(target_dir)):
        if not fname.endswith(".npy"):
            continue
        fpath = os.path.join(target_dir, fname)
        try:
            _ = np.load(fpath)
        except Exception as e:
            print(f"[❌ ERRORE] {fpath} → {e}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mel_dir", required=True, help="Directory con file .npz")
    parser.add_argument("--target_dir", required=True, help="Directory con file .npy")
    args = parser.parse_args()

    verify_npz_files(args.mel_dir)
    verify_npy_files(args.target_dir)

if __name__ == "__main__":
    main()

