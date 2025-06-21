import json
import os
import yaml


def update_phoneme_mapping(phonemes, path, special_tokens=None):
    if special_tokens is None:
        special_tokens = ["<PAD>", "<SIL>", "<START>"]

    # Carica config per offset
    with open("config.yaml", "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    offset = config.get("special_tokens_offset", 10)

    # Inizializza mappa
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            phoneme_to_index = json.load(f)
    else:
        phoneme_to_index = {
            token: idx for idx, token in enumerate(special_tokens)
        }

    next_id = max(phoneme_to_index.values(), default=-1) + 1
    # Forza offset minimo
    if next_id < offset:
        next_id = offset

    updated = False
    for ph in phonemes:
        if ph not in phoneme_to_index:
            phoneme_to_index[ph] = next_id
            next_id += 1
            updated = True

    if updated:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(phoneme_to_index, f, ensure_ascii=False, indent=2)

    return phoneme_to_index

