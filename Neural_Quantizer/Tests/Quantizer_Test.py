import unittest
import numpy as np


from Algorithms.Quantizer import Quantizer


class Quantizer_Test(unittest.TestCase):


    def test_valid_int8_limits(self): # Input validation
        q = Quantizer([1.0, 2.0], -128, 127)
        self.assertEqual(q.q_min, -128)
        self.assertEqual(q.q_max, 127)


    def test_invalid_int8_limits(self):
        invalid_ranges = [(-129, 127), (-128, 128), (10, 10), (100, -100)]

        for q_min, q_max in invalid_ranges:
            with self.subTest(q_min=q_min, q_max=q_max):
                with self.assertRaises(ValueError):
                    Quantizer([1.0], q_min, q_max)


    def test_empty_input(self):
        with self.assertRaises(ValueError):
             Quantizer([])


    def test_complex_values(self):
        with self.assertRaises((TypeError)):
            Quantizer([1.0 + 2j], [1.0 + 2j])

    def test_non_numeric_input(self):
        with self.assertRaises(TypeError):
            Quantizer(["a", "b", "c"], 0.1, 0)


    def test_inf_input(self):
        with self.assertRaises(ValueError):
            Quantizer([1.0, np.inf, 2.0])


    def test_symmetric_scale(self): # Symmetric calibration tests
        q = Quantizer([-1.0, 0.0, 1.0])
        q.symmetric_calibrate()

        self.assertAlmostEqual(q.symmetric_scale, 1.0 / 127)


    def test_symmetric_custom_limits(self):
        q = Quantizer([-2.0, 0.0, 2.0], -100, 100)
        q.symmetric_calibrate()

        self.assertAlmostEqual(q.symmetric_scale, 2.0 / 100)


    def test_symmetric_rejects_asymmetric_limits(self):
        q = Quantizer([-1.0, 1.0], -128, 127)

        with self.assertRaises(ValueError):
            q.symmetric_calibrate()


    def test_symmetric_zero_data(self):
        q = Quantizer([0.0, 0.0, 0.0])
        q.symmetric_calibrate()

        self.assertEqual(q.symmetric_scale, 1.0)


    def test_asymmetric_scale(self): # Asymmetric calibration tests
        data = [-0.5, 0.1, 0.5, 2.0]

        q = Quantizer(data, -128, 127)
        q.asymmetric_calibrate()

        expected_scale = 2.5 / 255

        self.assertAlmostEqual(q.asymmetric_scale, expected_scale)


    def test_asymmetric_zero_point(self):
        data = [-0.5, 0.1, 0.5, 2.0]

        q = Quantizer(data, -128, 127)
        q.asymmetric_calibrate()

        self.assertEqual(q.zero_point, -77)


    def test_asymmetric_positive_data(self):
        q = Quantizer([0.1, 0.5, 1.0], -128, 127)
        q.asymmetric_calibrate()

        self.assertGreater(q.asymmetric_scale, 0)


    def test_asymmetric_negative_data(self):
        q = Quantizer([-2.0, -1.0, -0.1], -128, 127)
        q.asymmetric_calibrate()

        self.assertGreater(q.asymmetric_scale, 0)


    def test_asymmetric_zero_data(self):
        q = Quantizer([0.0, 0.0, 0.0])
        q.asymmetric_calibrate()

        self.assertEqual(q.asymmetric_scale, 1.0)
        self.assertIsNotNone(q.zero_point)


    def test_asymmetric_constant_data(self):
        q = Quantizer([2.0, 2.0, 2.0], -128, 127)
        q.asymmetric_calibrate()

        self.assertEqual(q.asymmetric_scale, 1.0)
        self.assertTrue(-128 <= q.zero_point <= 127)


    def test_symmetric_quantization(self):  # Symmetric quantization tests
        q = Quantizer([-1.0, 0.0, 1.0])

        result = q.symmetric_quantize()

        expected = np.array([-127, 0, 127], dtype=np.int8)

        np.testing.assert_array_equal(result, expected)


    def test_asymmetric_quantization(self): # Asymmetric quantization tests
        q = Quantizer([-0.5, 0.1, 0.5, 2.0], -128, 127)

        result = q.asymmetric_quantize()

        expected = np.array([-128, -67, -26, 127], dtype=np.int8)

        np.testing.assert_array_equal(result, expected)


    def test_symmetric_output_dtype(self): # Output validation
        q = Quantizer([-1.0, 0.0, 1.0])
        result = q.symmetric_quantize()

        self.assertEqual(result.dtype, np.int8)
        

    def test_asymmetric_output_dtype(self):
        q = Quantizer([-1.0, 0.0, 1.0])
        result = q.asymmetric_quantize()

        self.assertEqual(result.dtype, np.int8)


    def test_symmetric_output_range(self):
        q = Quantizer([-100.0, 0.0, 100.0])
        result = q.symmetric_quantize()

        self.assertTrue(np.all(result >= q.q_min))
        self.assertTrue(np.all(result <= q.q_max))
        

    def test_asymmetric_output_range(self):
        q = Quantizer([-100.0, 0.0, 100.0])
        result = q.asymmetric_quantize()

        self.assertTrue(np.all(result >= q.q_min))
        self.assertTrue(np.all(result <= q.q_max))


    def test_random_data(self): # Random data tests
        rng = np.random.default_rng(42)

        data = rng.normal(0, 1, 1000).astype(np.float32)

        q = Quantizer(data)

        symmetric = q.symmetric_quantize()
        asymmetric = q.asymmetric_quantize()

        self.assertEqual(symmetric.shape, data.shape)
        self.assertEqual(asymmetric.shape, data.shape)

        self.assertEqual(symmetric.dtype, np.int8)
        self.assertEqual(asymmetric.dtype, np.int8)


if __name__ == "__main__":
    unittest.main()
