import numpy as np


class C_Generator:


    @staticmethod
    def generate_c_file(quantized_data, scale, zero_point, filename):
        quantized_data = np.asarray(quantized_data)
        
        if quantized_data.size == 0:
            raise ValueError("Quantized data cannot be empty!")

        if not np.issubdtype(quantized_data.dtype, np.integer):
            raise TypeError("Quantized data must contain integers!")

        int8_min = np.iinfo(np.int8).min 
        int8_max = np.iinfo(np.int8).max 
        if quantized_data.max() > int8_max or quantized_data.min() < int8_min:
            raise ValueError("Quantized data must be in INT8 range!")

        if not np.isfinite(scale) or scale <= 0:
            raise ValueError("Scale must be positive and finite!")

        if not isinstance(zero_point, (int, np.integer)):
            raise TypeError("Zero point must be an integer!")

        if (zero_point < int8_min or zero_point > int8_max):
            raise ValueError("Zero point must be in INT8 range!")

        with open(filename, "w") as file:
            file.write("#include <stdint.h>\n\n\n")

            file.write("const int8_t quantized_data[] = {")

            values = ", ".join(map(str, quantized_data.flatten()))

            file.write(f"{values}")
            file.write("};\n\n\n")

            file.write(f"const unsigned int data_size = {quantized_data.size};\n\n")
            file.write(f"const float scale = {scale}f;\n")
            file.write(f"const int zero_point = {zero_point};\n")
