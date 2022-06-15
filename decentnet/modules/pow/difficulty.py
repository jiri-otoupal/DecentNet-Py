from dataclasses import dataclass


@dataclass
class Difficulty:
    tCost: int
    mCost: int
    pCost: int
    nBits: int
    hashLen: int

    def __post_init__(self):
        if (req_m_cost := 8 * self.pCost) > self.mCost:
            print(f"Memory too low increasing from {self.mCost} to {req_m_cost}")
            self.mCost = req_m_cost
