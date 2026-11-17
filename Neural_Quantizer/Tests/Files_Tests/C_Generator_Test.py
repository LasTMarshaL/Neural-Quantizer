import unittest
import numpy as np
import os


from Files.C_Generator import C_Generator


class TestCGenerator(unittest.TestCase):

   
    def test_empty_data(self): # Input validation tests
        with self.assertRaises(ValueError):
            C_Generator.generate_c_file(np.array([], dtype=np.int8), 0.025, "test_output.c", "Algorithms/Dequantizer.c", -5)


    def test_invalid_data_type(self):
        with self.assertRaises(TypeError):
            C_Generator.generate_c_file(np.array([1.5, 2.5, 3.5]), 0.025, "test_output.c", "Algorithms/Dequantizer.c", -5)
            

    def test_data_out_of_range(self):
        with self.assertRaises(ValueError):
            C_Generator.generate_c_file(np.array([128, -129]), 0.025, "test_output.c", "Algorithms/Dequantizer.c", -5)


    def test_invalid_scale(self):
        with self.assertRaises(ValueError):
           C_Generator.generate_c_file(np.array([12, -45, 100, -128], dtype=np.int8), -0.5, "test_output.c", "Algorithms/Dequantizer.c", -5)


    def test_invalid_zero_point(self):
        with self.assertRaises(FileNotFoundError):
            C_Generator.generate_c_file(np.array([12, -45, 100, -128], dtype=np.int8), -0.5, "test_output.c", "Non_File.c", -5)


    def test_invalid_zero_point(self):
        with self.assertRaises(ValueError):
            C_Generator.generate_c_file(np.array([12, -45, 100, -128], dtype=np.int8), -0.5, "test_output.c", "Algorithms/Dequantizer.c", 128)
    
    
    def test_file_content(self): # File creation and content test
        C_Generator.generate_c_file(np.array([12, -45, 100, -128], dtype=np.int8), 0.025, "test_output.c", "Algorithms/Dequantizer.c", -5)
    
        with open("test_output.c", "r") as file:
            content_c = file.read()

        with open("test_output.h", "r") as file:
            content_h = file.read()

        with open("Algorithms/Dequantizer.c", "r") as file:
            contend_d = file.read()

        self.assertTrue(os.path.exists("test_output.c"))
        self.assertTrue(os.path.exists("test_output.h"))
        
        self.assertIn("#include <stdint.h>", content_c)
        self.assertIn("12, -45, 100, -128", content_c)
        self.assertIn("const int zero_point = -5;", content_c)
        self.assertIn("const unsigned int data_size = 4;", content_c)

        self.assertIn("#ifndef QUANTIZED_DATA_H", content_h)
        self.assertIn("#define QUANTIZED_DATA_H", content_h)
        self.assertIn("#include <stdint.h>", content_h)
        self.assertIn("extern const int8_t quantized_data[];", content_h)
        self.assertIn("extern const unsigned int data_size;", content_h)
        self.assertIn("extern const float scale;", content_h)
        self.assertIn("extern const int zero_point;", content_h)
        self.assertIn("#endif", content_h)

        self.assertIn('#include "../test_output.h"', contend_d)
        

if __name__ == "__main__":
    unittest.main()