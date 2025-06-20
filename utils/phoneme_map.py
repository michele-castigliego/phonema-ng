import json
import os

def update_phoneme_mapping(phonemes, path):
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            phoneme_to_index = json.load(f)
    else:
        phoneme_to_index = {
            "<PAD>": 0,
            "<SIL>": 1,
            "<START>": 2
        }

    next_id = max(phoneme_to_index.values()) + 1
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

