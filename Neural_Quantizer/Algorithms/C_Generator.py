import numpy as np


class C_Generator:


    @staticmethod
    def generate_c_file(data, scale, zero_point, filename):
        data = np.asarray(data)
        
        if data.size == 0:
            raise ValueError("Input data cannot be empty!")

        if not np.issubdtype(data.dtype, np.integer):
            raise TypeError("Input data must contain integers!")

        if np.any(data < -128) or np.any(data > 127):
            raise ValueError("Input data must be in INT8 range!")

        if not np.isfinite(scale) or scale <= 0:
            raise ValueError("Scale must be positive and finite!")

        if not isinstance(zero_point, (int, np.integer)):
            raise TypeError("Zero point must be an integer!")

        if not -128 <= zero_point <= 127:
            raise ValueError("Zero point must be in INT8 range!")

        with open(filename, "w") as file:
            file.write("#include <stdint.h>\n\n\n")

            file.write("const int8_t quantized_data[] = {\n")

            values = ", ".join(map(str, data.flatten()))

            file.write(f"    {values}\n")
            file.write("};\n\n\n")

            file.write(f"const unsigned int data_size = {data.size};\n\n")
            file.write(f"const float scale = {scale}f;\n")
            file.write(f"const int zero_point = {zero_point};\n")
