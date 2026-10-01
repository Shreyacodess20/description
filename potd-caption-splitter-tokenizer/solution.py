import numpy as np

def tokenize(vocab: dict[str, int], caption: str) -> list[int]:
    words = caption.split()
    return [vocab.get(w, 1) for w in words]
