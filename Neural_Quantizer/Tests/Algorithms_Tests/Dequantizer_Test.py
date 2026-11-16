import unittest
import numpy as np


from Algorithms.Dequantizer import Dequantizer


class Dequantizer_Test(unittest.TestCase):


    def test_empty_input(self): # Input validation tests
        with self.assertRaises(ValueError):
            Dequantizer([], 0.1, 0)


    def test_non_numeric_input(self):
        with self.assertRaises(TypeError):
            Dequantizer(["a", "b", "c"], 0.1, 0)


    def test_complex_values(self):
        with self.assertRaises((TypeError)):
            Dequantizer([1.0 + 2j], [1.0 + 2j])


    def test_nan_input(self):
        with self.assertRaises(ValueError):
            Dequantizer([1, np.nan, 3], 0.1, 0)


    def test_inf_input(self):
        with self.assertRaises(ValueError):
            Dequantizer([1, np.inf, 3], 0.1, 0)

    def test_non_integer_values(self):
        with self.assertRaises(TypeError):
            Dequantizer([1.5, 2.5, 3.5], 0.1, 0)


    def test_out_of_int8_range(self):
        with self.assertRaises(ValueError):
            Dequantizer([-129, 128], 0.1, 0)


    def test_zero_scale(self): # Scale Validation
        with self.assertRaises(ValueError):
            Dequantizer([1, 2, 3], 0, 0)


    def test_negative_scale(self):
        with self.assertRaises(ValueError):
            Dequantizer([1, 2, 3], -0.1, 0)


    def test_nan_scale(self):
        with self.assertRaises(ValueError):
            Dequantizer([1, 2, 3], np.nan, 0)


    def test_inf_scale(self):
        with self.assertRaises(ValueError):
            Dequantizer([1, 2, 3], np.inf, 0)


    def test_non_numeric_scale(self):
        with self.assertRaises(TypeError):
            Dequantizer([1, 2, 3], "scale", 0)


    def test_non_integer_zero_point(self): # Zero-point Validation
        with self.assertRaises(TypeError):
            Dequantizer([1, 2, 3], 0.1, 1.5)


    def test_out_of_range_zero_point(self):
        with self.assertRaises(ValueError):
            Dequantizer([1, 2, 3], 0.1, 128)


    def test_negative_out_of_range_zero_point(self):
        with self.assertRaises(ValueError):
            Dequantizer([1, 2, 3], 0.1, -129)


    def test_symmetric_basic(self): # Symmetric Dequantization
        data = np.array([-127, -64, 0, 64, 127], dtype=np.int8)
        scale = 0.01

        dequantizer = Dequantizer(data, scale, 0)
        result = dequantizer.symmetric_dequantize()

        expected = np.array([-1.27, -0.64, 0.0, 0.64, 1.27], dtype=np.float32)

        np.testing.assert_allclose(result, expected)


    def test_symmetric_zero_values(self):
        data = np.array([0, 0, 0], dtype=np.int8)

        dequantizer = Dequantizer(data, 0.5, 0)
        result = dequantizer.symmetric_dequantize()

        np.testing.assert_array_equal(result, np.array([0.0, 0.0, 0.0], dtype=np.float32))


    def test_symmetric_output_dtype(self):
        data = np.array([10, 20, 30], dtype=np.int8)

        dequantizer = Dequantizer(data, 0.1, 0)
        result = dequantizer.symmetric_dequantize()

        self.assertEqual(result.dtype, np.float32)


    def test_symmetric_boundary_values(self):
        data = np.array([-128, 127], dtype=np.int8)

        dequantizer = Dequantizer(data, 1.0, 0)
        result = dequantizer.symmetric_dequantize()

        np.testing.assert_array_equal(result,np.array([-128.0, 127.0], dtype=np.float32))

 
    def test_asymmetric_basic(self): # Asymmetric Dequantization
        data = np.array([10, 20, 30], dtype=np.int8)
        scale = 0.5
        zero_point = 10

        dequantizer = Dequantizer(data, scale, zero_point)
        result = dequantizer.asymmetric_dequantize()

        expected = np.array([0.0, 5.0, 10.0], dtype=np.float32)

        np.testing.assert_allclose(result, expected)

    def test_asymmetric_zero_point(self):
        data = np.array([0, 10, 20], dtype=np.int8)
        scale = 0.1
        zero_point = 10

        dequantizer = Dequantizer(data, scale, zero_point)
        result = dequantizer.asymmetric_dequantize()

        expected = np.array([-1.0, 0.0, 1.0], dtype=np.float32)

        np.testing.assert_allclose(result, expected)


    def test_asymmetric_output_dtype(self):
        data = np.array([10, 20, 30], dtype=np.int8)

        dequantizer = Dequantizer(data, 0.1, 10)
        result = dequantizer.asymmetric_dequantize()

        self.assertEqual(result.dtype, np.float32)


    def test_asymmetric_boundary_zero_point(self):
        data = np.array([-128, 127], dtype=np.int8)

        dequantizer = Dequantizer(data, 1.0, -128)
        result = dequantizer.asymmetric_dequantize()

        expected = np.array([0.0, 255.0], dtype=np.float32)

        np.testing.assert_array_equal(result, expected)


if __name__ == "__main__":
    unittest.main()