import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from utils.positional_encoding import PositionalEncoding

class StreamingConformerBlock(keras.layers.Layer):
    def __init__(self, d_model, ff_dim, num_heads, dropout=0.1, causal=True):
        super().__init__()
        self.ff_pre = keras.Sequential([
            layers.LayerNormalization(),
            layers.Dense(ff_dim, activation="relu"),
            layers.Dropout(dropout),
            layers.Dense(d_model),
            layers.Dropout(dropout)
        ])

        self.attn_norm = layers.LayerNormalization()
        self.mha = layers.MultiHeadAttention(
            num_heads=num_heads,
            key_dim=d_model // num_heads,
            dropout=dropout
        )
        self.dropout_attn = layers.Dropout(dropout)

        self.conv_norm = layers.LayerNormalization()
        self.conv_proj = layers.Conv1D(filters=2 * d_model, kernel_size=1, activation="gelu")
        self.causal = causal
        self.depthwise_conv = layers.DepthwiseConv1D(
            kernel_size=15,
            padding="valid" if causal else "same"
        )
        self.batch_norm = layers.BatchNormalization()
        self.conv_out = layers.Conv1D(filters=d_model, kernel_size=1)
        self.dropout_conv = layers.Dropout(dropout)

        self.ff_post = keras.Sequential([
            layers.LayerNormalization(),
            layers.Dense(ff_dim, activation="relu"),
            layers.Dropout(dropout),
            layers.Dense(d_model),
            layers.Dropout(dropout)
        ])

    def call(self, x, cache=None, training=False):
        """Forward pass with optional caching.

        Parameters
        ----------
        x : tf.Tensor
            Input tensor of shape ``(B, T, d_model)`` containing only the new
            frames when ``cache`` is provided.
        cache : dict or None
            Dictionary with the following keys (all optional): ``"k"``, ``"v"``
            and ``"conv"``. They store past key/value tensors for the
            self-attention and the previous inputs to the convolution module.
        training : bool
            Training flag.

        Returns
        -------
        tf.Tensor or Tuple[tf.Tensor, dict]
            If ``cache`` is ``None`` the output tensor is returned. Otherwise a
            tuple ``(output, new_cache)`` is returned where ``new_cache``
            contains the updated tensors to be fed on the next call.
        """

        new_cache = None

        # === Feed Forward Pre ===
        x_ff = self.ff_pre(x, training=training)

        # === Self-Attention ===
        attn_input = self.attn_norm(x)

        if cache is None:
            attn_output = self.mha(
                attn_input,
                attn_input,
                use_causal_mask=self.causal,
                training=training,
            )
        else:
            prev_k = cache.get("k")
            prev_v = cache.get("v")

            q = self.mha._query_dense(attn_input)
            k = self.mha._key_dense(attn_input)
            v = self.mha._value_dense(attn_input)

            if prev_k is not None:
                k = tf.concat([prev_k, k], axis=1)
                v = tf.concat([prev_v, v], axis=1)

            # Split heads
            num_heads = self.mha.num_heads
            key_dim = self.mha.key_dim

            def split_heads(t):
                t = tf.reshape(t, [tf.shape(t)[0], tf.shape(t)[1], num_heads, key_dim])
                return tf.transpose(t, [0, 2, 1, 3])

            q = split_heads(q)
            k = split_heads(k)
            v = split_heads(v)

            # Scaled dot-product attention with causal mask that accounts for
            # cached timesteps.
            dk = tf.cast(key_dim, tf.float32)
            scores = tf.einsum("bhqd,bhkd->bhqk", q, k) / tf.math.sqrt(dk)

            if self.causal:
                q_len = tf.shape(q)[2]
                k_len = tf.shape(k)[2]
                mem_len = k_len - q_len
                q_pos = tf.range(q_len) + mem_len
                k_pos = tf.range(k_len)
                mask = q_pos[:, None] >= k_pos[None, :]
                mask = tf.reshape(mask, [1, 1, q_len, k_len])
                scores = tf.where(mask, scores, tf.fill(tf.shape(scores), -1e9))

            attn_weights = tf.nn.softmax(scores, axis=-1)
            attn_weights = self.dropout_attn(attn_weights, training=training)
            attn_context = tf.einsum("bhqk,bhkd->bhqd", attn_weights, v)

            # Combine heads
            attn_context = tf.transpose(attn_context, [0, 2, 1, 3])
            attn_context = tf.reshape(
                attn_context,
                [tf.shape(attn_context)[0], tf.shape(attn_context)[1], num_heads * key_dim],
            )
            attn_output = self.mha._output_dense(attn_context)

            new_cache = {
                "k": tf.transpose(k, [0, 2, 1, 3])
            }
            new_cache["k"] = tf.reshape(new_cache["k"], [tf.shape(k)[0], tf.shape(k)[2], num_heads * key_dim])
            new_cache["v"] = tf.transpose(v, [0, 2, 1, 3])
            new_cache["v"] = tf.reshape(new_cache["v"], [tf.shape(v)[0], tf.shape(v)[2], num_heads * key_dim])

            # We only care about the output for the new frames
            if prev_k is not None:
                attn_output = attn_output[:, -tf.shape(x)[1] :, :]

        attn_output = self.dropout_attn(attn_output, training=training)
        x = x + 0.5 * x_ff + attn_output

        # === Convolution Module ===
        conv = self.conv_norm(x)
        conv = self.conv_proj(conv)

        pad_len = self.depthwise_conv.kernel_size[0] - 1

        if cache is None or cache.get("conv") is None:
            if self.causal:
                conv_padded = tf.pad(conv, [[0, 0], [pad_len, 0], [0, 0]])
                conv_cache = conv[:, -pad_len:, :]
            else:
                conv_padded = conv
                conv_cache = None
        else:
            conv_padded = tf.concat([cache["conv"], conv], axis=1)
            conv_cache = conv_padded[:, -pad_len:, :]

        conv_out = self.depthwise_conv(conv_padded)
        if cache is not None and cache.get("conv") is not None:
            conv_out = conv_out[:, -tf.shape(x)[1] :, :]

        conv_out = self.batch_norm(conv_out, training=training)
        conv_out = tf.nn.swish(conv_out)
        conv_out = self.conv_out(conv_out)
        conv_out = self.dropout_conv(conv_out, training=training)

        x += conv_out

        # === Feed Forward Post ===
        x += self.ff_post(x, training=training)

        if cache is None:
            return x
        else:
            new_cache["conv"] = conv_cache
            return x, new_cache


class StreamingConformerModel(keras.Model):
    """Streaming Conformer stack that exposes caching."""

    def __init__(
        self,
        n_mels=80,
        n_classes=128,
        d_model=256,
        num_heads=4,
        ff_dim=512,
        num_layers=4,
        dropout=0.1,
        causal=True,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.input_proj = layers.Dense(d_model)
        self.pos_enc = PositionalEncoding(d_model)
        self.blocks = [
            StreamingConformerBlock(d_model, ff_dim, num_heads, dropout, causal)
            for _ in range(num_layers)
        ]
        self.out_dense = layers.Dense(n_classes, name="frame_logits")

    def call(self, inputs, cache=None, training=False):
        x = self.input_proj(inputs)
        x = self.pos_enc(x)

        new_caches = []
        return_cache = cache is not None
        if cache is None:
            cache = [None] * len(self.blocks)
        for blk, blk_cache in zip(self.blocks, cache):
            if blk_cache is None:
                x = blk(x, training=training)
            else:
                x, blk_cache = blk(x, cache=blk_cache, training=training)
            new_caches.append(blk_cache)

        logits = self.out_dense(x)

        if return_cache:
            return logits, new_caches
        return logits


def build_streaming_conformer_model(
    n_mels=80,
    n_classes=128,
    d_model=256,
    num_heads=4,
    ff_dim=512,
    num_layers=4,
    dropout=0.1,
    causal=True,
):
    return StreamingConformerModel(
        n_mels=n_mels,
        n_classes=n_classes,
        d_model=d_model,
        num_heads=num_heads,
        ff_dim=ff_dim,
        num_layers=num_layers,
        dropout=dropout,
        causal=causal,
        name="StreamingConformer",
    )

# Alias per compatibilità con codice esistente
build_phoneme_segmentation_model = build_streaming_conformer_model

