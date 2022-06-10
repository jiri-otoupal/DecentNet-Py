from dataclasses import dataclass


@dataclass
class Difficulty:
    tCost: int
    mCost: int
    pCost: int
    nBits: int
    hashLen: int
