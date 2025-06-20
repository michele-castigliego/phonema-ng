# utils/index_dataset.py

import os
import numpy as np
import tensorflow as tf
import pandas as pd

def load_sample(mel_path, target_path):
    mel = np.load(mel_path)["mel"]
    target = np.load(target_path)
    return mel, target

def create_tf_dataset_from_index(index_csv, mel_dir, target_dir, batch_size=32, shuffle=True):
    df = pd.read_csv(index_csv)
    audio_ids = df["audio_id"].tolist()
    mel_paths = [os.path.join(mel_dir, f"{aid}.npz") for aid in audio_ids]
    target_paths = [os.path.join(target_dir, f"{aid}.npy") for aid in audio_ids]

    def generator():
        for mel_path, target_path in zip(mel_paths, target_paths):
            mel, target = load_sample(mel_path, target_path)
            yield mel, target

    # Get mel dimension without loading all
    dummy_mel = np.load(mel_paths[0])["mel"]
    n_mels = dummy_mel.shape[1]

    output_signature = (
        tf.TensorSpec(shape=(None, n_mels), dtype=tf.float32),
        tf.TensorSpec(shape=(None,), dtype=tf.int32),
    )

    dataset = tf.data.Dataset.from_generator(generator, output_signature=output_signature)

    def preprocess(mel, target):
        mel = tf.convert_to_tensor(mel, dtype=tf.float32)
        target = tf.convert_to_tensor(target, dtype=tf.int32)
        return mel, target

    dataset = dataset.map(preprocess)

    if shuffle:
        dataset = dataset.shuffle(buffer_size=len(mel_paths))

    dataset = dataset.padded_batch(batch_size, padded_shapes=([None, n_mels], [None]))
    return dataset

