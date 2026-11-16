import matplotlib.pyplot as plt
import numpy as np
import os


class Graphs_Constructor:


    def plot_dequantization(self, ax, original_data, dequantized_data_python, dequantized_data_c):
        ax.plot(original_data, label="Original Data", linewidth=1.5)
        ax.plot(dequantized_data_python, label="Dequantized Data Python", linewidth=1)
        ax.plot(dequantized_data_c, label="Dequantized Data C", linewidth=1)

        ax.set_title("Original VS Dequantized Data (Python VS C)")
        ax.set_xlabel("Index")
        ax.set_ylabel("Value")
        ax.legend()
        ax.grid()


    def plot_metrics(self, ax, metrics_python, metrics_c):
        names = ["MAE", "MSE", "RMSE", "Max Error"]

        python_values = [metrics_python["mae"], metrics_python["mse"], metrics_python["rmse"], metrics_python["max_error"]]
        c_values = [metrics_c["mae"], metrics_c["mse"], metrics_c["rmse"], metrics_c["max_error"]]

        column_groups_number = np.arange(len(names))
        column_width = 0.35

        ax.bar(column_groups_number - column_width / 2, python_values, column_width, label="Python")
        ax.bar(column_groups_number + column_width / 2, c_values, column_width, label="C")

        ax.set_title("Metrics Comparison")
        ax.set_ylabel("Error")
        ax.set_xticks(column_groups_number)
        ax.set_xticklabels(names)
        ax.legend()
        ax.grid()


    def visualize_quantization(self, original_data, dequantized_data_python, dequantized_data_c, metrics_python, metrics_c, saving_path=None):
        original_data = np.asarray(original_data)
        dequantized_data_python = np.asarray(dequantized_data_python)
        dequantized_data_c = np.asarray(dequantized_data_c)

        if original_data.size == 0 or dequantized_data_python.size == 0 or dequantized_data_c.size == 0:
            raise ValueError("Input data cannot be empty!")

        if original_data.shape != dequantized_data_python.shape or original_data.shape != dequantized_data_c.shape:
            raise ValueError("All input arrays must have equal shapes!")

        if not np.issubdtype(original_data.dtype, np.number) or not np.issubdtype(dequantized_data_python.dtype, np.number) or not np.issubdtype(dequantized_data_c.dtype, np.number):
            raise TypeError("All input arrays must contain numeric values!")

        if np.iscomplexobj(original_data) or np.iscomplexobj(dequantized_data_python) or np.iscomplexobj(dequantized_data_c):
            raise TypeError("Complex numbers are not supported!")

        if not np.all(np.isfinite(original_data)) or not np.all(np.isfinite(dequantized_data_python)) or not np.all(np.isfinite(dequantized_data_c)):
            raise ValueError("Input data contains NaN or Inf!")

        float32_max = np.finfo(np.float32).max
        float32_min = np.finfo(np.float32).min
        if any(data.max() > float32_max or data.min() < float32_min for data in [original_data, dequantized_data_python, dequantized_data_c]):
            raise ValueError("Input data must be in FLOAT32 range!")

        if not isinstance(metrics_python, dict) or not isinstance(metrics_c, dict):
            raise TypeError("Metrics must be dictionaries!")

        required_metrics = ["mae", "mse", "rmse", "max_error"]
        for metrics in [metrics_python, metrics_c]:
            if not all(metric in metrics for metric in required_metrics):
                raise ValueError("Metrics dictionary is missing required keys!")

            if not all(np.isscalar(metrics[metric]) and np.isfinite(metrics[metric]) for metric in required_metrics):
                raise ValueError("Metrics must contain finite scalar values!")

            if any(metrics[metric] < 0 for metric in required_metrics):
                raise ValueError("Metrics cannot be negative!")

        if saving_path is not None and not isinstance(saving_path, (str, bytes, os.PathLike)):
            raise TypeError("Saving path must be a string, bytes or PathLike object!")

        figure, axes = plt.subplots(1, 2, figsize=(14, 6))

        self.plot_dequantization(axes[0], original_data, dequantized_data_python, dequantized_data_c)
        self.plot_metrics(axes[1], metrics_python, metrics_c)

        figure.suptitle("INT8 Quantization Analysis", fontsize=16)
        figure.tight_layout()

        if saving_path is not None:
            figure.savefig(saving_path, dpi=150, bbox_inches="tight")

        plt.show()
        plt.close(figure)