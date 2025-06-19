import argparse
import numpy as np
import matplotlib.pyplot as plt
import librosa.display
import os
import pandas as pd

def plot_mel_with_phonemes(mel, phonemes, save_path=None, title=None):
    """
    Visualizza un mel-spectrogramma con i fonemi allineati sul tempo.

    Args:
        mel (np.ndarray): Matrice (T, 80) del mel-spectrogramma.
        phonemes (List[str]): Sequenza di fonemi da visualizzare.
        save_path (str): Se fornito, salva la figura invece di mostrarla.
        title (str): Titolo opzionale della figura.
    """
    fig, ax = plt.subplots(figsize=(12, 6))

    im = ax.imshow(mel.T, aspect="auto", origin="lower", interpolation="none")

    ax.set_ylabel("Mel bands")
    ax.set_xlabel("Time frames")

    if title:
        ax.set_title(title)

    # Annotazioni fonemi (opzionale: equi-distribuiti)
    T = mel.shape[0]
    num_phonemes = len(phonemes)
    step = T / num_phonemes

    for i, phon in enumerate(phonemes):
        center = int((i + 0.5) * step)
        if center < T:
            ax.text(center, mel.shape[1] + 2, phon, ha="center", va="bottom", fontsize=9, rotation=90, color="black")

    fig.colorbar(im, ax=ax, orientation='vertical')
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path)
        print(f"✅ Salvato: {save_path}")
        plt.close()
    else:
        plt.show()

def main(args):
    df = pd.read_csv(args.index_csv)
    row = df[df['id'] == args.id]
    if row.empty:
        print(f"❌ ID '{args.id}' non trovato in {args.index_csv}")
        return

    mel_path = row.iloc[0]['mel_path']
    data = np.load(mel_path, allow_pickle=True)
    mel = data['mel']
    phonemes = data['phonemes'].tolist()

    print(f"🔹 ID: {args.id}")
    print(f"🔸 Mel shape: {mel.shape}")
    print(f"🔸 N fonemi: {len(phonemes)}")

    plot_mel_with_phonemes(
        mel,
        phonemes,
        args.hop_length,
        args.sr,
        title=f"ID: {args.id}",
        save_path=args.save
    )

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Ispeziona mel-spectrogramma e fonemi")
    parser.add_argument("--index_csv", required=True)
    parser.add_argument("--id", required=True)
    parser.add_argument("--sr", type=int, default=16000)
    parser.add_argument("--hop_length", type=int, default=160)
    parser.add_argument("--save", type=str, default=None, help="Path per salvare PNG (opzionale)")
    args = parser.parse_args()
    main(args)

