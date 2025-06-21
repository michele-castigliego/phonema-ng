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
