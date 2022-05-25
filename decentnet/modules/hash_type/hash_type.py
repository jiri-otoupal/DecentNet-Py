import argon2

from decentnet.modules.pow.difficulty import Difficulty


class HashType:
    def __init__(self, diff: Difficulty, data: bytes):
        self.diff = diff
        self.value = self.hash(data)

    def hash(self, data: bytes):
        return argon2.hash_password_raw(data, None, self.diff.tCost, self.diff.mCost,
                                        self.diff.pCost, self.diff.hashLen)

    def recompute(self, data: bytes):
        self.value = self.hash(data)

    def value_as_int(self) -> int:
        return int.from_bytes(self.value, "big")
