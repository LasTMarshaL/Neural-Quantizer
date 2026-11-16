import unittest
import numpy as np
import tempfile
import os


from Files.Data_Loader import DataLoader


class TestDataLoader(unittest.TestCase):


    def test_load_npy(self):
        expected = np.array([1.0, 2.0, 3.0])

        with tempfile.TemporaryDirectory() as temporary_directory:
            filepath = os.path.join(temporary_directory, "data.npy")
            np.save(filepath, expected)

            result = DataLoader.load_data(filepath)

            np.testing.assert_array_equal(result, expected)


    def test_file_not_found(self):
        with self.assertRaises(FileNotFoundError):
            DataLoader.load_data("nonexistent.npy")


    def test_unsupported_extension(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            filepath = os.path.join(temporary_directory, "data.json")

            with self.assertRaises(ValueError):
                DataLoader.load_data(filepath)


if __name__ == "__main__":
    unittest.main()