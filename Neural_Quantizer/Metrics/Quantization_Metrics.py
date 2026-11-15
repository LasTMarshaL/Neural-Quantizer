import numpy as np


class QuantizationMetrics:

        def __init__(self, original_data, restored_data):
            input_original_data = np.asarray(original_data, dtype=np.float32)
            input_restored_data = np.asarray(restored_data, dtype=np.float32)

            if input_original_data.size == 0 or input_restored_data.size == 0:
                raise ValueError("Input data cannot be empty!")

            if input_original_data.shape != input_restored_data.shape:
                raise ValueError("Original and restored data must have equal shapes!")

            if not np.issubdtype(input_original_data.dtype, np.number) or not np.issubdtype(input_restored_data.dtype, np.number):
                raise TypeError("Original data must contain numeric values!")

            if np.iscomplexobj(input_original_data) or np.iscomplexobj(input_restored_data):
                raise TypeError("Complex numbers are not supported!")

            if not np.all(np.isfinite(input_original_data)) or not np.all(np.isfinite(input_restored_data)):
                raise ValueError("Original data contains NaN or Inf!")

            self.original_data = input_original_data
            self.restored_data = input_restored_data
            self.error = self.original_data - self.restored_data

             
        def mae(self):
            return np.mean(np.abs(self.error))


        def mse(self):
            return np.mean(np.square(self.error))


        def rmse(self):
            return np.sqrt(np.mean(np.square(self.error)))


        def max_error(self):
            return np.max(np.abs(self.error))