import numpy as np


class Dequantizer:


    def __init__(self, quantized_data, scale, zero_point=0):
        quantized_data = np.asarray(quantized_data)

        if quantized_data.size == 0:
            raise ValueError("Quantized data cannot be empty!")

        if not np.issubdtype(quantized_data.dtype, np.number):
            raise TypeError("Quantized data must contain numeric values!")

        if np.iscomplexobj(quantized_data):
            raise TypeError("Complex numbers are not supported!")
        
        if not np.all(np.isfinite(quantized_data)):
            raise ValueError("Quantized contains NaN or Inf!")

        if not np.issubdtype(quantized_data.dtype, np.integer):
            raise TypeError("Quantized data must contain integers!")

        if quantized_data.min() < np.iinfo(np.int8).min or quantized_data.max() > np.iinfo(np.int8).max:
            raise ValueError("Quantized data must be in INT8 range!")

        if not isinstance(scale, (int, float, np.integer, np.floating)):
            raise TypeError("Scale must be numeric!")

        if not np.isfinite(scale) or scale <= 0:
            raise ValueError("Scale must be finite and positive!")

        if not isinstance(zero_point, (int, np.integer)):
            raise TypeError("Zero-point must be an integer!")

        if not -128 <= zero_point <= 127:
            raise ValueError("Zero point must be in INT8 range!")

        self.quantized_data = quantized_data.astype(np.int8)
        self.scale = float(scale)
        self.zero_point = zero_point


    def symmetric_dequantize(self):
        return self.quantized_data.astype(np.float32) * self.scale


    def asymmetric_dequantize(self):
        return (self.quantized_data.astype(np.float32) - self.zero_point) * self.scale