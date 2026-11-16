import unittest
import matplotlib
matplotlib.use("Agg")


import matplotlib.pyplot as plt
import numpy as np
import tempfile
import os


from Visualization.Graphs_Constructor import Graphs_Constructor


class Graphs_Constructor_Test(unittest.TestCase):


    def test_empty_data(self): # Input validation tests
        with self.assertRaises(ValueError):    
            Graphs_Constructor().visualize_quantization(
                [],
                [1.1, 2.1, 2.9, 4.1],
                [1.0, 2.0, 3.1, 4.0],
                {"mae": 0.1, "mse": 0.01, "rmse": 0.1, "max_error": 0.1},
                {"mae": 0.05, "mse": 0.005, "rmse": 0.07, "max_error": 0.1}
            )


    def test_different_shapes(self):
        with self.assertRaises(ValueError):  
            Graphs_Constructor().visualize_quantization(
                [1],
                [1.1, 2.1, 2.9, 4.1],
                [1.0, 2.0, 3.1, 4.0],
                {"mae": 0.1, "mse": 0.01, "rmse": 0.1, "max_error": 0.1},
                {"mae": 0.05, "mse": 0.005, "rmse": 0.07, "max_error": 0.1}
            )


    def test_non_numeric_data(self):
        with self.assertRaises(TypeError):      
            Graphs_Constructor().visualize_quantization(
                ["a", "b", "c", "d"],
                [1.1, 2.1, 2.9, 4.1],
                [1.0, 2.0, 3.1, 4.0],
                {"mae": 0.1, "mse": 0.01, "rmse": 0.1, "max_error": 0.1},
                {"mae": 0.05, "mse": 0.005, "rmse": 0.07, "max_error": 0.1}
            )


    def test_complex_data(self):
        with self.assertRaises(TypeError):     
            Graphs_Constructor().visualize_quantization(
                [1+2j, 2+3j, 3+4j, 4+5j],
                [1.1, 2.1, 2.9, 4.1],
                [1.0, 2.0, 3.1, 4.0],
                {"mae": 0.1, "mse": 0.01, "rmse": 0.1, "max_error": 0.1},
                {"mae": 0.05, "mse": 0.005, "rmse": 0.07, "max_error": 0.1}
            )


    def test_nan_data(self):
        with self.assertRaises(ValueError):
            Graphs_Constructor().visualize_quantization(
                [1.0, np.nan, 3.0, 4.0],
                [1.1, 2.1, 2.9, 4.1],
                [1.0, 2.0, 3.1, 4.0],
                {"mae": 0.1, "mse": 0.01, "rmse": 0.1, "max_error": 0.1},
                {"mae": 0.05, "mse": 0.005, "rmse": 0.07, "max_error": 0.1}
            )


    def test_inf_data(self):
        with self.assertRaises(ValueError): 
            Graphs_Constructor().visualize_quantization(
                [1.0, 2.0, 3.0, 4.0],
                [1.1, 2.1, np.inf, 4.1],
                [1.0, 2.0, 3.1, 4.0],
                {"mae": 0.1, "mse": 0.01, "rmse": 0.1, "max_error": 0.1},
                {"mae": 0.05, "mse": 0.005, "rmse": 0.07, "max_error": 0.1}
            )


    def test_float32_range(self):
        with self.assertRaises(ValueError):
            Graphs_Constructor().visualize_quantization(
                [1.0, 2.0, 3.0, 3.5e38],
                [1.1, 2.1, 2.9, 4.1],
                [1.0, 2.0, 3.1, 4.0],
                {"mae": 0.1, "mse": 0.01, "rmse": 0.1, "max_error": 0.1},
                {"mae": 0.05, "mse": 0.005, "rmse": 0.07, "max_error": 0.1}
            )


    def test_invalid_metrics_type(self):
        with self.assertRaises(ValueError):
            Graphs_Constructor().visualize_quantization(
                [1.0, 2.0, 3.0, 3.5e38],
                [1.1, 2.1, 2.9, 4.1],
                [1.0, 2.0, 3.1, 4.0],
                [0.1, 0.01, 0.1, 0.1],
                {"mae": 0.05, "mse": 0.005, "rmse": 0.07, "max_error": 0.1}
            )


    def test_missing_metric(self):
        with self.assertRaises(ValueError):
            Graphs_Constructor().visualize_quantization(
                [1.0, 2.0, 3.0, 4.0],
                [1.1, 2.1, 2.9, 4.1],
                [1.0, 2.0, 3.1, 4.0],
                {"mae": 0.1, "mse": 0.01, "rmse": 0.1, "max_error": 0.1},
                {"mae": 0.005, "rmse": 0.07, "max_error": 0.1}
        )


    def test_negative_metric(self):
        with self.assertRaises(ValueError):
            Graphs_Constructor().visualize_quantization(
                [1.0, 2.0, 3.0, 4.0],
                [1.1, 2.1, 2.9, 4.1],
                [1.0, 2.0, 3.1, 4.0],
                {"mae": 0.1, "mse": 0.01, "rmse": -0.1, "max_error": 0.1},
                {"mae": 0.05, "mse": 0.005, "rmse": 0.07, "max_error": 0.1}
        )


    def test_invalid_saving_path(self):
        with self.assertRaises(TypeError):
            Graphs_Constructor().visualize_quantization(
                [1.0, 2.0, 3.0, 4.0],
                [1.1, 2.1, 2.9, 4.1],
                [1.0, 2.0, 3.1, 4.0],
                {"mae": 0.1, "mse": 0.01, "rmse": 0.1, "max_error": 0.1},
                {"mae": 0.05, "mse": 0.005, "rmse": 0.07, "max_error": 0.1},
                234
        )


    def test_visualize_quantization(self): # Visualization tests
        plt.show = lambda: None

        Graphs_Constructor().visualize_quantization(
            [1.0, 2.0, 3.0, 4.0],
            [1.1, 2.1, 2.9, 4.1],
            [1.0, 2.0, 3.1, 4.0],
            {"mae": 0.1, "mse": 0.01, "rmse": 0.1, "max_error": 0.1},
            {"mae": 0.05, "mse": 0.005, "rmse": 0.07, "max_error": 0.1}
        )


    def test_save_figure(self):
        plt.show = lambda: None

        with tempfile.TemporaryDirectory() as directory:
            saving_path = os.path.join(directory, "test_graph.png")
 
            Graphs_Constructor().visualize_quantization(
                [1.0, 2.0, 3.0, 4.0],
                [1.1, 2.1, 2.9, 4.1],
                [1.0, 2.0, 3.1, 4.0],
                {"mae": 0.1, "mse": 0.01, "rmse": 0.1, "max_error": 0.1},
                {"mae": 0.05, "mse": 0.005, "rmse": 0.07, "max_error": 0.1},
                saving_path
            )

            self.assertTrue(os.path.exists(saving_path))
            self.assertGreater(os.path.getsize(saving_path), 0)


if __name__ == "__main__":
    unittest.main()