#cython: language_level=3
import numpy as np

from decentnet.modules.convert.byte_operations import int_to_bytes


def compute_pow(bits, hash_t, nonce, value):
    while np.log2(float(value)) > bits:
        value = hash_t.value_as_int()
        hash_t.recompute(
            int_to_bytes((value + nonce)).tobytes())
        nonce += 1
    return nonce