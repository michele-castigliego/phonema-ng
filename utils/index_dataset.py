# utils/index_dataset.py

import os
import pandas as pd
import numpy as np
import tensorflow as tf

def create_tf_dataset_from_index(index_csv_path, mel_dir, target_dir,
                                 batch_size=32, shuffle=False, shuffle_seed=42,
                                 max_frames=None):
    index_df = pd.read_csv(index_csv_path)
    audio_ids = index_df["audio_id"].tolist()
    mel_paths = [os.path.join(mel_dir, f"{aid}.npz") for aid in audio_ids]
    target_paths = [os.path.join(target_dir, f"{aid}.npy") for aid in audio_ids]

    if shuffle:
        rng = np.random.default_rng(seed=shuffle_seed)
        indices = rng.permutation(len(audio_ids))
        mel_paths = [mel_paths[i] for i in indices]
        target_paths = [target_paths[i] for i in indices]

    def generator():
        for mel_path, target_path in zip(mel_paths, target_paths):
            mel = np.load(mel_path)["mel"]
            if max_frames is not None and mel.shape[0] > max_frames:
                continue
            target = np.load(target_path)
            yield mel, target

    # Trova primo campione valido
    first_valid_idx = next(i for i, path in enumerate(mel_paths)
                           if max_frames is None or np.load(path)["mel"].shape[0] <= max_frames)
    sample_mel = np.load(mel_paths[first_valid_idx])["mel"]
    n_mels = sample_mel.shape[1]

    output_signature = (
        tf.TensorSpec(shape=(None, n_mels), dtype=tf.float32),
        tf.TensorSpec(shape=(None,), dtype=tf.int32),
    )

    dataset = tf.data.Dataset.from_generator(generator, output_signature=output_signature)

    dataset = dataset.padded_batch(
        batch_size,
        padded_shapes=([None, n_mels], [None]),
        padding_values=(0.0, 0),
        drop_remainder=False
    )

    return dataset
