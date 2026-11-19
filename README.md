## Neural-Quantizer
A Python-based quantization and dequantization project with C integration, designed to explore and compare symmetric and asymmetric quantization methods.

## 📌About
The project implements quantization algorithms in Python, generates C source and header files containing quantized data and its parameters, and uses GCC to compile C code into shared libraries. These libraries are then loaded and executed directly from Python using ctypes.

## 🗂️ Repository Structure
* **`Neural_Quantizer/`** - All code and implementations.
* **`Neural_Quantizer/Algorithms/`** - Quantization and dequantization algorithms. ⚠️ **!Attention!**: Dequantizer.c can contain syntax error, until the user launched the code, because .c file containing data has not been generated.
* **`Neural_Quantizer/Files/`** - Files loading and generation.  
* **`Neural_Quantizer/Metrics/`** - Quantization metrics calculation.
* **`Neural_Quantizer/Tests/`** - Tests (categories are separated into subfolders).
* **`Neural_Quantizer/Visualization/`** - Graphs plotting and analysis.

## 💻 Short Software Overview
* **Development environment**: Visual Studio Code
* **Programming Languages**: Python, C
* **C tools and libraries**: GCC 16.2.0 was used to compile C code and the folowing libaries were used: stdint.h, math.h, stddef.h, assert.h, stdio.h
* **Python tools and libraries**: The following libraries were used: numpy, os, matplotlib, ctypes, subprocess, sys, unittest
    
## ⚠️ Safety Notice
⚠️!The author bears no responsibility for the reader's actions!⚠️

## 🧪 Used Technologies & Software
*  Visual Studio Code (Development environment)
* C (Dequantization algorithm & Tests)
* Python (Math, Algorithms, Files, Metrics, Visualization + Tests)

## Description and Implementation

# 1) Idea:
This project was created to implement quantization and dequantization (float32 ↔ int8) and also metrics collection and visualization. Quantization is implemented with Python, however dequantization is implemented by Python and C. Metrics collection and visualization allows to compare received results graphically. All blocks were tested.

# 2) Algorithms:
Quantization and dequantization can be symmetric and asymmetric (based on formulas that can be found in the Internet) using numpy, while dequantization in C is implemented with math.h and stdint.h. Another point is thatdata for quantization are read from a data file with data (more details in Files), while C functions receive input data from a generated file, which was included as a library (more details in Files).

# 3) Files:
Data loading is implemented with numpy and os. Supported extensions: .npy, .csv, .txt . C file generation is also implemented with numpy and os. it creates .c file with quantized data and .h file for use it in other .c files.

# 4) Metrics:
This project part calculates: MAE, MSE, RMSE and MAX_ERROR (based on quantizad and dequantized data) for analysis.

# 5) Visualization:
Visualization is implemented with matplotlib, numpy and os. It draws graphs (based on qunatized and dequantized data + metrics) to see the defferences and compare C and Python dequntization results.

# 6) Tests:
All blocks and codes were checked by Python unittest and C assert.h, to verify execution and exception raising.

# 7) Main:
The main file is a Python script that connects all the modules described above and writes the results to a .txt file. In addition to NumPy and os, the sys, ctypes, and subprocess libraries are used. sys is responsible for identifying the user's operating system (Windows / Linux / macOS), while ctypes and subprocess are used to connect C and Python. Specifically, subprocess executes a compilation command to compile the generated .c files into a shared library, whereas ctypes loads the compiled library and provides access to C functions and their parameters (pointers) from the Python script. Finally, two graphs are generated (comparing symmetric and asymmetric dequantization in C and Python), and the results of the metric calculations are written to a generated .txt file.

# 8) Conclusion:
As a result, the data dequantized by Python and C correspond almost precisely to each other and to the original data (Data.txt). The two graphs below illustrate the results graphically, while the metric calculation results are also written to Result.txt.
<img width="1751" height="738" alt="image" src="https://github.com/user-attachments/assets/44d792b2-ce4a-49f7-b547-f497abaa02c5" />
<img width="1562" height="663" alt="image" src="https://github.com/user-attachments/assets/3a2d3450-24ea-41bf-9afa-8d0c58f30dbe" />

