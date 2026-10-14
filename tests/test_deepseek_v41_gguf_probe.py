import struct
import unittest

from tools.deepseek_v41_gguf_probe import parse_header


def string(value):
    return struct.pack("<Q", len(value)) + value


class GGUFProbeTest(unittest.TestCase):
    def fixture(self):
        return (b"GGUF" + struct.pack("<IQQ", 3, 1, 1) + string(b"general.architecture") +
                struct.pack("<I", 8) + string(b"deepseek4") + string(b"tensor.weight") +
                struct.pack("<IQQIQ", 2, 32, 8, 0, 0))

    def test_minimal_header(self):
        metadata, tensors, offset = parse_header(self.fixture())
        self.assertEqual(metadata["general.architecture"], "deepseek4")
        self.assertEqual(tensors[0]["shape"], [32, 8])
        self.assertEqual(offset % 32, 0)

    def test_truncated_prefix(self):
        with self.assertRaises(EOFError):
            parse_header(self.fixture()[:-1])

    def test_wrong_magic(self):
        with self.assertRaises(ValueError):
            parse_header(b"BAD!" + self.fixture()[4:])

    def test_bad_alignment(self):
        data = b"GGUF" + struct.pack("<IQQ", 3, 0, 1) + string(b"general.alignment") + struct.pack("<II", 4, 3)
        with self.assertRaises(ValueError):
            parse_header(data)


if __name__ == "__main__":
    unittest.main()
