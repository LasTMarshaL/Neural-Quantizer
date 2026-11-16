import numpy as np


class Quantization_Metrics:

        def __init__(self, original_data, dequantized_data):
            original_data = np.asarray(original_data)
            dequantized_data = np.asarray(dequantized_data)

            if original_data.size == 0 or dequantized_data.size == 0:
                raise ValueError("Input data cannot be empty!")

            if original_data.shape != dequantized_data.shape:
                raise ValueError("Original and dequantized data must have equal shapes!")

            if not np.issubdtype(original_data.dtype, np.number) or not np.issubdtype(dequantized_data.dtype, np.number):
                raise TypeError("Original data must contain numeric values!")

            if np.iscomplexobj(original_data) or np.iscomplexobj(dequantized_data):
                raise TypeError("Complex numbers are not supported!")

            if not np.all(np.isfinite(original_data)) or not np.all(np.isfinite(dequantized_data)):
                raise ValueError("Original data contains NaN or Inf!")

            float32_max = np.finfo(np.float32).max
            float32_min = np.finfo(np.float32).min
            if np.any((original_data > float32_max) | (original_data < float32_min)) or np.any((dequantized_data > float32_max) | (dequantized_data < float32_min)):
                raise ValueError("Input data must be in FLOAT32 range!")

            self.original_data = original_data.astype(np.float32)
            self.restored_data = dequantized_data.astype(np.float32)
            self.error = self.original_data - self.restored_data

             
        def mae(self) -> float: 
            return np.mean(np.abs(self.error))


        def mse(self) -> float:
            return np.mean(np.square(self.error))


        def rmse(self) -> float:
            return np.sqrt(np.mean(np.square(self.error)))


        def max_error(self) -> float:
            return np.max(np.abs(self.error))