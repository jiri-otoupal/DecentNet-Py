import unittest

from decentnet.modules.pow.difficulty import Difficulty

from decentnet.modules.pow.pow import PoW


class TestPOW(unittest.TestCase):
    def test_basic(self):
        bits_diff = 16
        diff = Difficulty(1, 8, 1, bits_diff, 32)
        pw = PoW(0x121524564513212315231, diff).compute()
        print(f"Difficulty {bits_diff}")
        print(pw.finished_nonce)
        as_int = pw.finished_hash.value_as_int()
        print(f"{as_int :#0{64}x}")
        print('0x{0:0{1}x}'.format(as_int, 32))


if __name__ == '__main__':
    unittest.main()
