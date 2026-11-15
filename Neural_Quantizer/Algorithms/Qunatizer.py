import numpy as np


class Quantizer:


    def __init__(self, data, q_min=-127, q_max=127):
        input_data = np.asarray(data, dtype=np.float32)

        if self.data.size == 0:
            raise ValueError("Input data cannot be empty!")

        if not np.issubdtype(input_data.dtype, np.number):
            raise ValueError("Input data must contain numeric values!")
        
        if not np.all(np.isfinite(input_data)):
            raise ValueError("Input data contains NaN or Inf!")
        
        if not (-128 <= q_min < q_max <= 127):
            raise ValueError("Invalid INT8 quantization range!")

        self.data = input_data
        self.q_max = q_max
        self.q_min = q_min
        self.zero_point = None
        self.symmetric_scale = None
        self.asymmetric_scale = None
        self.symmetric_quantized = None
        self.asymmetric_quantized = None


    def symmetric_calibrate(self):
        if -self.q_min != self.q_max:
            raise ValueError("Symmetric quantization requires symmetric limits!")
        
        max_value = np.max(np.abs(self.data))
        self.symmetric_scale = max_value / self.q_max if (self.q_max != 0 and max_value != 0) else 1.0


    def asymmetric_calibrate(self):
        max_value = np.max(self.data)
        min_value = np.min(self.data)
        self.asymmetric_scale = (max_value - min_value) / (self.q_max - self.q_min) if min_value != max_value else 1.0

        self.zero_point = int(np.clip(np.round(self.q_min - min_value / self.asymmetric_scale), self.q_min, self.q_max))


    def symmetric_quantize(self):
        if self.symmetric_scale is None:
            self.symmetric_calibrate()

        q = np.clip(np.round(self.data / self.symmetric_scale), self.q_min, self.q_max)
        return q.astype(np.int8)


    def asymmetric_quantize(self):
        if self.asymmetric_scale is None or self.zero_point is None:
            self.asymmetric_calibrate()
    
        q = np.clip(np.round(self.data / self.asymmetric_scale) + self.zero_point, self.q_min, self.q_max)
        return q.astype(np.int8)