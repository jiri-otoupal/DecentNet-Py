import time
from typing import Optional

import pyximport

pyximport.install()

from decentnet.modules.pow.computation import compute_pow  # noqa

from decentnet.modules.hash_type.hash_type import HashType
from decentnet.modules.pow.difficulty import Difficulty


class PoW:
    def __init__(self, input_hash: int, diff: Difficulty):
        self.finished_nonce: Optional[int] = None
        self.finished_hash: Optional[HashType] = None
        self.previous_hash = input_hash
        self.difficulty = diff
        self.compute_time = None

    def compute(self):
        nonce = 0
        hash_t = HashType(self.difficulty,
                          self.previous_hash.to_bytes(self.difficulty.hashLen,
                                                      byteorder='big'))
        bits = 256 - self.difficulty.nBits
        value = hash_t.value_as_int()

        nonce = compute_pow(bits, hash_t, nonce, value)

        self.finished_hash = hash_t
        self.finished_nonce = nonce

        return self
