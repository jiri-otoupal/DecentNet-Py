import unittest

from decentnet.modules.pow.difficulty import Difficulty
from decentnet.modules.pow.pow import PoW


class TestPOW(unittest.TestCase):
    def test_basic(self):
        for _ in range(8):
            bits_diff = 16
            diff = Difficulty(1, 8, 1, bits_diff, 32)
            pw = PoW(0x121524564513212315231, diff).compute()
            print(f"Difficulty {bits_diff}")
            print(pw.finished_nonce)
            padded = pw.finished_hash.value_as_hex()
            self.assertEqual(padded[:2], "00")
            print(f"0x{padded}")


if __name__ == '__main__':
    unittest.main()
