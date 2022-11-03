import glob

import numpy as np
import scipy.io as sp
import re
import matplotlib.pyplot as plt

data_set = []
def get_data_per_pixel(directory_path):
    HSI_images_list=glob.glob(directory_path)
    for HSIAddress in HSI_images_list:
        HSI_mat_file = sp.loadmat(HSIAddress)
        HSI = list(HSI_mat_file.keys())[-1]  # usually the last key of the dictonary is the name of the file
        HSI = np.asarray(HSI_mat_file[HSI])  # save the image content as a 3D array
        HSI_size = HSI.shape  # get the size of the array
        if len(HSI_size) != 3:
            continue
        # Pre Processing - prepare the data set in pixel driven batch. for each pixel take all the layers
        data = []  # generate data set for each pixel, a set of fixed X,Y - take all the layers combined as a vector
        for x in range(0, HSI_size[0]):
            for y in range(0, HSI_size[1]):
                data.append(HSI[x, y, :])  # for each X,Y - take all the laywers as vector
        data_set.append(np.array(data))  # convert it to numpy array


get_data_per_pixel("/Users/mayagoldman/PycharmProjects/Moshal-Hackathon/data/*")
print(data_set)