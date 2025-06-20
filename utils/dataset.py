# utils/dataset.py

import os
import json
import numpy as np
import tensorflow as tf

def load_sample(mel_path, target_path):
    mel = np.load(mel_path)["mel"]
    target = np.load(target_path)
    return mel, target

def create_tf_dataset(mel_dir, target_dir, batch_size=32, shuffle=True):
    mel_files = [f for f in os.listdir(mel_dir) if f.endswith(".npz")]
    mel_paths = [os.path.join(mel_dir, f) for f in mel_files]
    target_paths = [os.path.join(target_dir, f.replace(".npz", ".npy")) for f in mel_files]

    # Carica un file per dedurre il numero di mel bins
    example_mel = np.load(mel_paths[0])["mel"]
    feature_dim = example_mel.shape[1]

    def generator():
        for mel_path, target_path in zip(mel_paths, target_paths):
            mel, target = load_sample(mel_path, target_path)
            yield mel, target

    def preprocess(mel, target):
        mel = tf.convert_to_tensor(mel, dtype=tf.float32)
        target = tf.convert_to_tensor(target, dtype=tf.int32)
        return mel, target

    output_signature = (
        tf.TensorSpec(shape=(None, feature_dim), dtype=tf.float32),
        tf.TensorSpec(shape=(None,), dtype=tf.int32),
    )

    dataset = tf.data.Dataset.from_generator(generator, output_signature=output_signature)
    dataset = dataset.map(preprocess)

    if shuffle:
        dataset = dataset.shuffle(buffer_size=len(mel_paths))

    dataset = dataset.padded_batch(batch_size, padded_shapes=([None, feature_dim], [None]))
    return dataset

