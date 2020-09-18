import csv
import numpy as np
from PIL import Image
from numpy import loadtxt
import time

class data_helper:
    def __init__(self):
        self.training_set = []
        self.training_labels = []
        self.training_size = 0
        self.test_set = []

    def load_training_data(self, path):
        with open(path) as csv_file:
            csv_reader = csv.reader(csv_file, delimiter=',')
            for row in csv_reader:
                self.training_set.append([int(i) for i in row[1:]])
                self.training_labels.append(int(row[0]))

        assert len(self.training_set) == len(self.training_labels), "data loaded incorrectly"
        self.training_size = len(self.training_set)

    def load_testing_data(self, path):
        with open(path) as csv_file:
            csv_reader = csv.reader(csv_file, delimiter=',')
            for row in csv_reader:
                self.test_set.append([int(i) for i in row])

    def reshape_array(self, array):
        assert len(array) == 28*28, "image has incorrect dimensions"
        reordered_pixels = [array[28*i:28*(i+1)] for i in range(0,28)]
        return np.array(reordered_pixels, dtype=np.uint8)

    def visualise_train_data(self, row_number):
        assert row_number < len(self.training_set), "dataset size smaller than row number given"
        label = self.training_labels[row_number]
        pixels = self.training_set[row_number]
        pixel_image = self.reshape_array(pixels)
        img = Image.fromarray(pixel_image, 'L')
        img.show()
        print(label)

    def visualise_test_data(self, row_number):
        assert row_number < len(self.test_set), "dataset size smaller than row number given"
        pixels = self.test_set[row_number]
        pixel_image = self.reshape_array(pixels)
        img = Image.fromarray(pixel_image, 'L')
        img.show()
        time.sleep(2)
        img.close()

"""
DH = data_helper()
DH.load_testing_data("test_temp.csv")

DH.visualise_test_data(27998)
DH.visualise_test_data(27999)
DH.visualise_test_data(28000)
"""
