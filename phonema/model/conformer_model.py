import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from utils.positional_encoding import PositionalEncoding


def build_phoneme_segmentation_model(
    n_mels=80,
    n_classes=128,
    d_model=256,
    num_heads=4,
    ff_dim=512,
    num_layers=4,
    dropout=0.1,
    l1=0.0,
    l2=0.0,
    causal=False
):
    inputs = keras.Input(shape=(None, n_mels), name="mel_spectrogram")

    # Dense projection + Positional Encoding
    x = layers.Dense(d_model)(inputs)
    x = PositionalEncoding(d_model)(x)

    # Conformer Encoder Blocks
    for _ in range(num_layers):
        # Feed Forward Module (pre)
        ff_pre = keras.Sequential([
            layers.LayerNormalization(),
            layers.Dense(ff_dim, activation="relu"),
            layers.Dropout(dropout),
            layers.Dense(d_model),
            layers.Dropout(dropout)
        ])
        x1 = ff_pre(x)

        # Multi-head Self Attention
        attn = layers.LayerNormalization()(x)
        mha = layers.MultiHeadAttention(
            num_heads=num_heads,
            key_dim=d_model // num_heads,
            dropout=dropout
        )
        attn = mha(attn, attn, use_causal_mask=causal)
        attn = layers.Dropout(dropout)(attn)
        x2 = x + 0.5 * x1 + attn  # Combine FF-pre and attention

        # Convolution Module
        conv = layers.LayerNormalization()(x2)
        conv = layers.Conv1D(filters=2 * d_model, kernel_size=1, activation="gelu")(conv)

        # DepthwiseConv1D with causal padding (if needed)
        if causal:
            pad_len = 15 - 1  # kernel_size - 1
            conv = layers.ZeroPadding1D(padding=(pad_len, 0))(conv)
            conv = layers.DepthwiseConv1D(
                kernel_size=15,
                padding="valid"
            )(conv)
        else:
            conv = layers.DepthwiseConv1D(
                kernel_size=15,
                padding="same"
            )(conv)

        conv = layers.BatchNormalization()(conv)
        conv = layers.Activation("swish")(conv)
        conv = layers.Conv1D(filters=d_model, kernel_size=1)(conv)
        conv = layers.Dropout(dropout)(conv)
        x3 = x2 + conv

        # Feed Forward Module (post)
        ff_post = keras.Sequential([
            layers.LayerNormalization(),
            layers.Dense(ff_dim, activation="relu"),
            layers.Dropout(dropout),
            layers.Dense(d_model),
            layers.Dropout(dropout)
        ])
        x4 = x3 + 0.5 * ff_post(x3)

        # Residual
        x = layers.LayerNormalization()(x4)

    # Output classifier with L1/L2 regularization
    outputs = layers.Dense(
        n_classes,
        activation="softmax",
        name="phoneme_class",
        kernel_regularizer=tf.keras.regularizers.L1L2(l1=l1, l2=l2)
    )(x)

    return keras.Model(inputs=inputs, outputs=outputs)

