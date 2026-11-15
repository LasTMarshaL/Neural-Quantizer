import unittest
import numpy as np
import os


from Algorithms.C_Generator import C_Generator


class TestCGenerator(unittest.TestCase):

   
    def test_empty_data(self): # Input validation tests
        with self.assertRaises(ValueError):
            C_Generator.generate_c_file(np.array([], dtype=np.int8), 0.025, -5, "test_output.c")


    def test_invalid_data_type(self):
        with self.assertRaises(TypeError):
            C_Generator.generate_c_file(np.array([1.5, 2.5, 3.5]), 0.025, -5, "test_output.c")
            

    def test_data_out_of_range(self):
        with self.assertRaises(ValueError):
            C_Generator.generate_c_file(np.array([128, -129]), 0.025, -5, "test_output.c")


    def test_invalid_scale(self):
        with self.assertRaises(ValueError):
           C_Generator.generate_c_file(np.array([12, -45, 100, -128], dtype=np.int8), -0.5, -5, "test_output.c")


    def test_invalid_zero_point(self):
        with self.assertRaises(ValueError):
            C_Generator.generate_c_file(np.array([12, -45, 100, -128], dtype=np.int8), -0.5, 128, "test_output.c")
    
    
    def test_file_creation(self): # File creation tests
        C_Generator.generate_c_file(np.array([12, -45, 100, -128], dtype=np.int8), 0.025, -5, "test_output.c")
    
        self.assertTrue(os.path.exists("test_output.c"))
    
    
    def test_file_content(self):
        C_Generator.generate_c_file(np.array([12, -45, 100, -128], dtype=np.int8), 0.025, -5, "test_output.c")
    
        with open("test_output.c", "r") as file:
            content = file.read()
    
        self.assertIn("#include <stdint.h>", content)
        self.assertIn("12, -45, 100, -128", content)
        self.assertIn("const int zero_point = -5;", content)
        self.assertIn("const unsigned int data_size = 4;", content)


if __name__ == "__main__":
    unittest.main()