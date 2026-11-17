import numpy as np
import os


class C_Generator:


    @staticmethod
    def generate_c_file(quantized_data, scale, generated_filename, dequantizer_filename, zero_point=0):
        quantized_data = np.asarray(quantized_data)
        
        if quantized_data.size == 0:
            raise ValueError("Quantized data cannot be empty!")

        if not np.issubdtype(quantized_data.dtype, np.integer):
            raise TypeError("Quantized data must contain integers!")

        if quantized_data.min() < np.iinfo(np.int8).min or quantized_data.max() > np.iinfo(np.int8).max:
            raise ValueError("Quantized data must be in INT8 range!")

        if not np.isfinite(scale) or scale <= 0:
            raise ValueError("Scale must be positive and finite!")

        if not os.path.exists(dequantizer_filename):
            raise FileNotFoundError("Dequantizer C file does not exists!")

        if not isinstance(zero_point, (int, np.integer)):
            raise TypeError("Zero point must be an integer!")

        if (zero_point < np.iinfo(np.int8).min  or zero_point > np.iinfo(np.int8).max):
            raise ValueError("Zero point must be in INT8 range!")

        with open(os.path.splitext(generated_filename)[0] + ".h", "w") as file:
            file.write("#ifndef QUANTIZED_DATA_H\n")
            file.write("#define QUANTIZED_DATA_H\n\n\n")

            file.write("#include <stdint.h>\n\n\n")

            file.write("extern const int8_t quantized_data[];\n")
            file.write("extern const unsigned int data_size;\n")
            file.write("extern const float scale;\n")
            file.write("extern const int zero_point;\n\n\n")

            file.write("#endif\n")

        with open(generated_filename, "w") as file:
            file.write("#include <stdint.h>\n\n\n")

            file.write("const int8_t quantized_data[] = {")

            values = ", ".join(map(str, quantized_data.flatten()))

            file.write(f"{values}")
            file.write("};\n\n\n")

            file.write(f"const unsigned int data_size = {quantized_data.size};\n\n")
            file.write(f"const float scale = {scale}f;\n")
            file.write(f"const int zero_point = {zero_point};\n")

        include = f'#include "../{os.path.splitext(os.path.basename(generated_filename))[0]}.h"\n'

        with open(dequantizer_filename, "r") as file:
            lines = file.readlines()

        if not include in lines:

            lines.insert(4, include + "\n")

            with open(dequantizer_filename, "w") as file:
                file.writelines(lines)