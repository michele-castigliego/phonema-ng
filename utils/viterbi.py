import json
import os
import numpy as np


def viterbi_decode(logits: np.ndarray, transition_matrix: np.ndarray) -> np.ndarray:
    """Decode the most probable state sequence using the Viterbi algorithm.

    Parameters
    ----------
    logits : np.ndarray
        Array of shape ``(T, num_states)`` containing unnormalized log
        probabilities for each frame.
    transition_matrix : np.ndarray
        Square matrix ``(num_states, num_states)`` where ``A[i, j]`` is the
        probability of transitioning from state ``i`` to state ``j``.

    Returns
    -------
    np.ndarray
        Sequence of state indices corresponding to the most probable path.
    """
    # Convert emissions to log-probabilities
    log_probs = logits - logits.max(axis=1, keepdims=True)
    log_probs = log_probs - np.log(np.exp(log_probs).sum(axis=1, keepdims=True))

    # Convert transition matrix to log space, avoiding log(0)
    with np.errstate(divide="ignore"):
        log_trans = np.log(transition_matrix)
    log_trans[~np.isfinite(log_trans)] = -np.inf

    num_frames, num_states = log_probs.shape
    dp = np.full((num_frames, num_states), -np.inf, dtype=np.float32)
    backpointers = np.zeros((num_frames, num_states), dtype=np.int32)

    # Initialization
    dp[0] = log_probs[0]

    # Dynamic programming
    for t in range(1, num_frames):
        for j in range(num_states):
            prev_scores = dp[t - 1] + log_trans[:, j]
            best_prev = np.argmax(prev_scores)
            dp[t, j] = prev_scores[best_prev] + log_probs[t, j]
            backpointers[t, j] = best_prev

    # Backtracking
    best_last_state = int(np.argmax(dp[-1]))
    best_path = [best_last_state]
    for t in range(num_frames - 1, 0, -1):
        best_last_state = backpointers[t, best_last_state]
        best_path.append(best_last_state)

    best_path.reverse()
    return np.array(best_path, dtype=np.int32)


def build_viterbi_matrix(
    phonemized_dir: str,
    phoneme_map_path: str,
    lang: str,
    output_dir: str,
    splits=("train", "dev", "test"),
) -> np.ndarray:
    """Create or update the transition matrix used by :func:`viterbi_decode`.

    Parameters
    ----------
    phonemized_dir : str
        Directory containing ``phonemized_<split>.jsonl`` files.
    phoneme_map_path : str
        Path to ``phoneme_to_index.json`` mapping file.
    lang : str
        Language code used to name the output matrix file.
    output_dir : str
        Directory where the resulting ``viterbi-matrix_<lang>.npy`` is saved.
    splits : tuple of str, optional
        Dataset splits to parse. Defaults to ``("train", "dev", "test")``.

    Returns
    -------
    np.ndarray
        The computed transition probability matrix.
    """
    with open(phoneme_map_path, "r", encoding="utf-8") as f:
        phoneme_to_index = json.load(f)

    num_classes = len(phoneme_to_index)

    matrix_path = os.path.join(output_dir, f"viterbi-matrix_{lang}.npy")
    if os.path.exists(matrix_path):
        A = np.load(matrix_path)
    else:
        A = np.zeros((num_classes, num_classes), dtype=np.float32)

    counts = np.zeros_like(A)

    for split in splits:
        file_path = os.path.join(phonemized_dir, f"phonemized_{split}.jsonl")
        if not os.path.exists(file_path):
            continue
        with open(file_path, "r", encoding="utf-8") as f:
            for line in f:
                data = json.loads(line)
                phonemes = data.get("phonemes", [])
                ids = [phoneme_to_index[p] for p in phonemes if p in phoneme_to_index]
                for i in range(1, len(ids)):
                    A[ids[i - 1], ids[i]] += 1
                    counts[ids[i - 1], ids[i]] += 1

    for i in range(num_classes):
        total = np.sum(counts[i])
        if total > 0:
            A[i] = A[i] / total
        else:
            A[i] = 0
            A[i, i] = 1.0

    os.makedirs(output_dir, exist_ok=True)
    np.save(matrix_path, A)
    return A
