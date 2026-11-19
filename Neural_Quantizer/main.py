import numpy as np
import ctypes
import subprocess as sp
import os
import sys


from Algorithms.Quantizer import Quantizer
from Algorithms.Dequantizer import Dequantizer

from Files.Data_Loader import Data_Loader
from Files.C_Generator import C_Generator

from Metrics.Quantization_Metrics import Quantization_Metrics
from Visualization.Visualizator import Visualizator


# Filepathes for compilations and generations
dequntizer_c_path = "Algorithms/Dequantizer.c"
generated_c_path = "Quantized.c"
symmetric_visualization_saving_path = "Symmetric_Visualization.png"
asymmetric_visualization_saving_path = "Asymmetric_Visualization.png"


# OS checking
if sys.platform.startswith("win"):
    libary_name_1 = "dequantizer_symmetric.dll"
    libary_name_2 = "dequantizer_asymmetric.dll"
elif sys.platform == "darwin":
    libary_name_1 = "dequantizer_symmetric.dylib"
    libary_name_2 = "dequantizer_asymmetric.dylib"
else: 
    libary_name_1 = "dequantizer_symmetric.so"
    libary_name_2 = "dequantizer_asymmetric.so"


# C libary for Python building
def build_c_library(compile_command):
    result = sp.run(compile_command, capture_output=True, text=True)
    
    if result.returncode != 0:
        print("GCC STDERR:")
        print(result.stderr)

        print("GCC STDOUT:")
        print(result.stdout)

        raise RuntimeError("C compilation failed!")


# Data loading and quantization
data = Data_Loader.load_data("Data.txt")

symmetric_quantizer = Quantizer(data)
asymmetric_quantizer = Quantizer(data, np.iinfo(np.int8).min, np.iinfo(np.int8).max)

symmetric_quantized_data = symmetric_quantizer.symmetric_quantize()
asymmetric_quantized_data = asymmetric_quantizer.asymmetric_quantize()

with open("Results.txt", "w") as file:
    file.write(f"QUANTIZATION AND DEQUANTIZATION RESULTS:\n\n\n")


# Symmetric Python
symmetric_dequantizer_python = Dequantizer(symmetric_quantized_data, symmetric_quantizer.symmetric_scale)
symmetric_dequantized_data_python = symmetric_dequantizer_python.symmetric_dequantize()

symmetric_quantization_metrics_python = Quantization_Metrics(data, symmetric_dequantized_data_python)
symmetric_quantization_metrics_python_mae = symmetric_quantization_metrics_python.mae()
symmetric_quantization_metrics_python_mse = symmetric_quantization_metrics_python.mse()
symmetric_quantization_metrics_python_rmse = symmetric_quantization_metrics_python.rmse()
symmetric_quantization_metrics_python_max_error = symmetric_quantization_metrics_python.max_error()

with open("Results.txt", "a") as file:
    file.write(f"Symmetric Python result:\nOriginal:\n{data}\nDequantized:\n{symmetric_dequantized_data_python}\n")
    file.write(f"Symmetric Python maximum error: {symmetric_quantization_metrics_python_max_error}\n\n")


# Symmetric C
C_Generator.generate_c_file(symmetric_quantized_data, symmetric_quantizer.symmetric_scale, generated_c_path, dequntizer_c_path)

symmetric_compile_command = ["gcc", "-shared", "-fPIC", dequntizer_c_path, generated_c_path, "-o", libary_name_1]

build_c_library(symmetric_compile_command)

libary_path_1 = os.path.abspath(libary_name_1)
libary_1 = ctypes.CDLL(libary_path_1)

libary_1.symmetric_dequantize.argtypes = [ctypes.POINTER(ctypes.c_float)]
libary_1.symmetric_dequantize.restype = ctypes.c_int

symmetric_dequantized_data_c = np.zeros(symmetric_quantized_data.size, dtype=np.float32)

pointer_symmetric_dequantized_data_c = symmetric_dequantized_data_c .ctypes.data_as(ctypes.POINTER(ctypes.c_float))

symmetric_dequantizer_c = libary_1.symmetric_dequantize(pointer_symmetric_dequantized_data_c)

if symmetric_dequantizer_c != 0:
    raise RuntimeError("Symmetric C dequantization failed!")

symmetric_quantization_metrics_c = Quantization_Metrics(data, symmetric_dequantized_data_c)
symmetric_quantization_metrics_c_mae = symmetric_quantization_metrics_c.mae()
symmetric_quantization_metrics_c_mse = symmetric_quantization_metrics_c.mse()
symmetric_quantization_metrics_c_rmse = symmetric_quantization_metrics_c.rmse()
symmetric_quantization_metrics_c_max_error = symmetric_quantization_metrics_c.max_error()

with open("Results.txt", "a") as file:
    file.write(f"Symmetric C result: \nOriginal:\n{data}\nDequantized:\n{symmetric_dequantized_data_c }\n")
    file.write(f"Symmetric C maximum error: {symmetric_quantization_metrics_c_max_error}\n\n")


# Asymmetric Python
asymmetric_dequantizer_python = Dequantizer(asymmetric_quantized_data, asymmetric_quantizer.asymmetric_scale, asymmetric_quantizer.zero_point)
asymmetric_dequantized_data_python = asymmetric_dequantizer_python.asymmetric_dequantize()

asymmetric_quantization_metrics_python = Quantization_Metrics(data, asymmetric_dequantized_data_python)
asymmetric_quantization_metrics_python_mae = asymmetric_quantization_metrics_python.mae()
asymmetric_quantization_metrics_python_mse = asymmetric_quantization_metrics_python.mse()
asymmetric_quantization_metrics_python_rmse = asymmetric_quantization_metrics_python.rmse()
asymmetric_quantization_metrics_python_max_error = asymmetric_quantization_metrics_python.max_error()

with open("Results.txt", "a") as file:
    file.write(f"Asymmetric Python result: \nOriginal:\n{data}\nDequantized:\n{asymmetric_dequantized_data_python}\n")
    file.write(f"Asymmetric Python maximum error: {asymmetric_quantization_metrics_python_max_error}\n\n")


# Asymmetric C
C_Generator.generate_c_file(asymmetric_quantized_data, asymmetric_quantizer.asymmetric_scale, generated_c_path, dequntizer_c_path, asymmetric_quantizer.zero_point)

asymmetric_compile_command = ["gcc", "-shared", "-fPIC", dequntizer_c_path, generated_c_path, "-o", libary_name_2]

build_c_library(asymmetric_compile_command)

libary_path_2 = os.path.abspath(libary_name_2)
libary_2 = ctypes.CDLL(libary_path_2)

libary_2.asymmetric_dequantize.argtypes = [ctypes.POINTER(ctypes.c_float)]
libary_2.asymmetric_dequantize.restype = ctypes.c_int

asymmetric_dequantized_data_c = np.zeros(asymmetric_quantized_data.size, dtype=np.float32)
pointer_asymmetric_dequantized_data_c = asymmetric_dequantized_data_c.ctypes.data_as(ctypes.POINTER(ctypes.c_float))

asymmetric_dequantizer_c = libary_2.asymmetric_dequantize(pointer_asymmetric_dequantized_data_c)

if asymmetric_dequantizer_c!= 0:
    raise RuntimeError("Asymmetric C dequantization failed!")

asymmetric_quantization_metrics_c = Quantization_Metrics(data, asymmetric_dequantized_data_c)
asymmetric_quantization_metrics_c_mae = asymmetric_quantization_metrics_c.mae()
asymmetric_quantization_metrics_c_mse = asymmetric_quantization_metrics_c.mse()
asymmetric_quantization_metrics_c_rmse = asymmetric_quantization_metrics_c.rmse()
asymmetric_quantization_metrics_c_max_error = asymmetric_quantization_metrics_c.max_error()

with open("Results.txt", "a") as file:
    file.write(f"Asymmetric C result: \nOriginal:\n{data}\nDequantized:\n{asymmetric_dequantized_data_c }\n")
    file.write(f"Asymmetric C maximum error: {asymmetric_quantization_metrics_c_max_error}\n\n")


# Visualization
Visualizator.visualize_quantization(data,
                                    symmetric_dequantized_data_python, 
                                    symmetric_dequantized_data_c,
                                    {"mae": symmetric_quantization_metrics_c_mae, 
                                    "mse": symmetric_quantization_metrics_python_mse, 
                                    "rmse": symmetric_quantization_metrics_python_rmse,
                                    "max_error": symmetric_quantization_metrics_python_max_error
                                    },
                                    {"mae": symmetric_quantization_metrics_c_mae,
                                    "mse": symmetric_quantization_metrics_c_mse,
                                    "rmse": symmetric_quantization_metrics_c_rmse,
                                    "max_error": symmetric_quantization_metrics_c_max_error
                                    },
                                    symmetric_visualization_saving_path
                                    )

Visualizator.visualize_quantization(data,
                                    asymmetric_dequantized_data_python, 
                                    asymmetric_dequantized_data_c,
                                    {"mae": asymmetric_quantization_metrics_c_mae, 
                                    "mse": asymmetric_quantization_metrics_python_mse, 
                                    "rmse": asymmetric_quantization_metrics_python_rmse,
                                    "max_error": asymmetric_quantization_metrics_python_max_error
                                    },
                                    {"mae": asymmetric_quantization_metrics_c_mae,
                                    "mse": asymmetric_quantization_metrics_c_mse,
                                    "rmse": asymmetric_quantization_metrics_c_rmse,
                                    "max_error": asymmetric_quantization_metrics_c_max_error
                                    },
                                    asymmetric_visualization_saving_path
                                    )