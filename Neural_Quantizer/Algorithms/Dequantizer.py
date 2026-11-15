import numpy as np


class Dequantizer:


    def __init__(self, quantized_data, scale, zero_point=0):
        input_data = np.asarray(quantized_data, np.int8)

        if input_data.size == 0:
            raise ValueError("Quantized  data cannot be empty!")

        if not np.issubdtype(input_data.dtype, np.number):
            raise ValueError("Quantized data must contain numeric values!")

        if not np.all(np.isfinite(input_data)):
            raise ValueError("Quantized contains NaN or Inf!")

        if not np.all(input_data == np.round(input_data)):
            raise ValueError("Quantized data must contain integers!")

        if np.any((input_data < -128) | (input_data > 127)):
            raise ValueError("Quantized data exceeds INT8 range!")

        if not isinstance(scale, (int, float, np.integer, np.floating)):
            raise ValueError("Scale must be numeric!")

        if not np.isfinite(scale) or scale <= 0:
            raise ValueError("Scale must be finite and positive!")

        if not isinstance(zero_point, (int, np.integer)):
            raise ValueError("Zero-point must be an integer!")

        if not -128 <= zero_point <= 127:
            raise ValueError("Zero-point exceeds INT8 range!")

        self.quantized_data = input_data.astype(np.int8)
        self.scale = float(scale)
        self.zero_point = zero_point


    def symmetric_dequantize(self):
        return self.quantized_data.astype(np.float32) * self.scale


    def asymmetric_dequantize(self):
        return (self.quantized_data.astype(np.float32) - self.zero_point) * self.scale