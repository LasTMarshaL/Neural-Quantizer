import numpy as np
import os


class Data_Loader:


    @staticmethod
    def load_data(filepath):
        if not os.path.exists(filepath):
            raise FileNotFoundError("Fail is not found!")
        
        extension = os.path.splitext(filepath)[1].lower()
        if (extension == ".npy"):
            data = np.load(filepath)
        elif(extension == ".csv"):
            data = np.loadtxt(filepath, delimiter=",")
        elif(extension == ".txt"):
            data = np.loadtxt(filepath)
        else:
            raise ValueError("Not supported file extension!")

        return data