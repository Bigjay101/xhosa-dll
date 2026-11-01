"""P10.3: matched random letter edit negative-control intervention (T13)."""
from __future__ import annotations

import random
import string
from src.segment import words


def add_letter_noise(text: str, per_word_rate: float, seed: int = 13) -> tuple[str, int]:
    """Edit each eligible word with the given probability using one a-z edit.

    Every attempted edit actually changes the lexical core. A deterministic
    seed makes the control reproducible. Punctuation is not edited.
    """
    if not 0 <= per_word_rate <= 1:
        raise ValueError("per_word_rate must be in [0,1]")
    rng = random.Random(seed)
    output, changed = [], 0
    for word in words(text):
        positions = [i for i, char in enumerate(word) if char in string.ascii_letters]
        if not positions or rng.random() >= per_word_rate:
            output.append(word)
            continue
        i = rng.choice(positions)
        ops = ["sub", "ins"] + (["del"] if len(positions) > 1 else [])
        op = rng.choice(ops)
        if op == "sub":
            alphabet = string.ascii_uppercase if word[i].isupper() else string.ascii_lowercase
            new_letter = rng.choice([a for a in alphabet if a != word[i]])
            rewritten = word[:i] + new_letter + word[i+1:]
        elif op == "ins":
            rewritten = word[:i] + rng.choice(string.ascii_lowercase) + word[i:]
        else:
            rewritten = word[:i] + word[i+1:]
        output.append(rewritten)
        changed += 1
    return " ".join(output), changed
