#cython: language_level=3
from math import log2

import argon2
import numpy as np

from decentnet.modules.convert.byte_operations import int_to_bytes

def compute_pow(bits, hash_t, nonce, value):
    while log2(value) >= bits:
        value = hash_t.value_as_int()
        hash_t.recompute(
            int_to_bytes((value + nonce)).tobytes())
        nonce += 1
    return nonce

def hash_func(data: bytes, diff):
    return argon2.hash_password_raw(data, None, diff.tCost, diff.mCost,
                                    diff.pCost, diff.hashLen)
