from decentnet.modules.pow.computation import hash_func

from decentnet.modules.pow.difficulty import Difficulty


class HashType:
    def __init__(self, diff: Difficulty, data: bytes):
        self.diff: Difficulty = diff
        self.value: bytes = hash_func(data, diff)

    def recompute(self, data: bytes):
        self.value: bytes = hash_func(data, self.diff)

    def value_as_hex(self):
        return self.value.hex()[2:].zfill(self.diff.hashLen * 2)

    def value_as_int(self) -> int:
        return int.from_bytes(self.value, "big")
