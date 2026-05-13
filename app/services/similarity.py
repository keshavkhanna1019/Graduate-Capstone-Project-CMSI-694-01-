from typing import Sequence, Union

import numpy as np


def cosine_similarity(a: Union[Sequence[float], np.ndarray], b: Union[Sequence[float], np.ndarray]) -> float:
    """Cosine similarity in [-1, 1]; robust to list/tuple and mildly denormalized vectors."""
    va = np.asarray(a, dtype=np.float64).ravel()
    vb = np.asarray(b, dtype=np.float64).ravel()
    if va.size != vb.size or va.size == 0:
        return 0.0
    na = float(np.linalg.norm(va))
    nb = float(np.linalg.norm(vb))
    if na == 0.0 or nb == 0.0:
        return 0.0
    va = va / na
    vb = vb / nb
    s = float(np.dot(va, vb))
    return max(-1.0, min(1.0, s))
