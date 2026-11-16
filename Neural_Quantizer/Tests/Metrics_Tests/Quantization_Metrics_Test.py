import unittest
import numpy as np


from Metrics.Quantization_Metrics import Quantization_Metrics


class Quantization_Metrics_Test(unittest.TestCase):


    def test_empty_original_data(self): # Input validation tests
        with self.assertRaises(ValueError):
            Quantization_Metrics([], [1.0])


    def test_empty_restored_data(self):
        with self.assertRaises(ValueError):
            Quantization_Metrics([1.0], [])


    def test_different_shapes(self):
        with self.assertRaises(ValueError):
            Quantization_Metrics([[1, 2], [3, 4]], [1, 2, 3, 4])


    def test_non_numeric_data(self):
        with self.assertRaises((TypeError)):
            Quantization_Metrics(["hello", "world"], [1.0, 2.0])


    def test_complex_values(self):
        with self.assertRaises((TypeError)):
             Quantization_Metrics([1.0 + 2j], [1.0 + 2j])


    def test_nan_values(self):
        with self.assertRaises(ValueError):
            Quantization_Metrics([1.0, np.nan], [1.0, 2.0])


    def test_inf_values(self):
        with self.assertRaises(ValueError):
            Quantization_Metrics([1.0, np.inf], [1.0, 2.0])


    def test_wrong_float32_range_values(self):
        with self.assertRaises(ValueError):
            Quantization_Metrics([1.0, 3.5e38], [1.0, 2.0])


    def test_mae(self): # MAE tests
        metrics = Quantization_Metrics([1.0, 2.0, 3.0, 4.0], [1.1, 1.8, 3.2, 3.9])

        self.assertAlmostEqual(metrics.mae(), 0.15)


    def test_mae_identical_arrays(self):
        metrics = Quantization_Metrics([1.0, 2.0, 3.0], [1.0, 2.0, 3.0])

        self.assertEqual(metrics.mae(), 0.0)


    def test_mse(self): # MSE tests
        metrics = Quantization_Metrics([1.0, 2.0, 3.0, 4.0], [1.1, 1.8, 3.2, 3.9])

        self.assertAlmostEqual(metrics.mse(), 0.025)


    def test_mse_identical_arrays(self):
        metrics = Quantization_Metrics([1.0, 2.0, 3.0], [1.0, 2.0, 3.0])

        self.assertEqual(metrics.mse(), 0.0)


    def test_rmse(self): # RMSE tests
        metrics = Quantization_Metrics([1.0, 2.0, 3.0, 4.0], [1.1, 1.8, 3.2, 3.9])

        self.assertAlmostEqual(metrics.rmse(), np.sqrt(0.025))


    def test_mse_identical_arrays(self):
            metrics = Quantization_Metrics([1.0, 2.0, 3.0], [1.0, 2.0, 3.0])
    
            self.assertEqual(metrics.rmse(), 0.0)


    def test_max_error(self): # Max Error tests
        metrics = Quantization_Metrics([1.0, 2.0, 3.0, 4.0], [1.1, 1.8, 3.2, 3.9])

        self.assertAlmostEqual(metrics.max_error(), 0.2)


    def test_max_error_negative_difference(self):
        metrics = Quantization_Metrics([1.0, 2.0, 3.0], [2.0, 1.0, 3.0])

        self.assertEqual(metrics.max_error(), 1.0)


if __name__ == "__main__":
    unittest.main()