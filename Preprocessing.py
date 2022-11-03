import glob

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy
from matplotlib.backends.backend_pdf import PdfPages
from sklearn.cluster import KMeans
from spectral import *
class HSI_processing():
    shape = None
    num_samples = None
    num_lines = None
    num_bands = None
    HSlayers = [5, 35, 50, 84]
    DataSet = []
    kmeans = None
    ClusteredPixels = []
    Segmentations = [2]
    mean_vecs = []

    def load_data(self, header_path):
        _image = envi.open(header_path)
        _data = _image.load()
        self.shape = _data.shape
        self.num_lines = _data.shape[0]
        self.num_samples = _data.shape[1]
        self.num_bands = _data.shape[2]

    def visualize_raw_data(self , header_path):
            _image = envi.open(header_path)
            _data = _image.load()
            for ChosenLayer in self.HSlayers:
                fig, ax = plt.subplots(figsize=(10, 7))
                plt.title(
                    f"ImageSize:{self.num_lines}x{self.num_samples} - {self.num_bands} Layers \n Layer:{ChosenLayer}")
                ImageNP = np.asarray(_data[:, :, ChosenLayer])  # take the 2D matrix of the chosen layer
                ax.set(xlabel='Xaxis', ylabel='Yaxis')
                plt.imshow(ImageNP)  # show it
                plt.show()
                print(f"Finish image process  | size: {self.shape} | Layer:{ChosenLayer}")

    def KMeans(self , header_path):
        _image = envi.open(header_path)
        _data = _image.load()
        for x in range(0, self.num_lines):
            for y in range(0, self.num_samples):
                self.DataSet.append(_data.read_pixel(x,y))  # for each X,Y - take all the laywers as vector
        for Clusters in self.Segmentations:
            print(f"Training {Clusters} Clusters | size: {self.shape} | Layer:")
            self.kmeans = KMeans(n_clusters=Clusters)  # create kmean model with number of clusters
            self.kmeans.fit(self.DataSet)  # train the model on the data
            # Check for each x,y position the cluster it belongs
            ClusteredPixel = np.zeros((self.num_lines, self.num_samples), dtype=int)
            for x in range(0, self.num_lines):
                for y in range(0, self.num_samples):
                    ClusteredPixel[x, y] = self.kmeans.predict(_data[x, y, :].reshape(1, -1).astype(float))
            fig, ax = plt.subplots(figsize=(10, 7))
            plt.title(
                f"ImageSize:{self.num_lines}x{self.num_samples} - {self.num_bands} Layers \n AI Algorithm for segmentation "
                f"\n Image splited to {Clusters} different segmentation")
            self.ClusteredPixels.append(ClusteredPixel)
            ax.set(xlabel='Xaxis', ylabel='Yaxis')
            plt.imshow(ClusteredPixel)
            plt.show()

    def get_cluster_mean(self):
        for i in range(len(self.Segmentations)):
            for k in range(self.Segmentations[i]):
                mask = np.argwhere(self.ClusteredPixels[i] == k)
                sum = 0
                for m in mask:
                    sum += self.DataSet[m[0] * self.num_lines + m[1]]
                mean = sum / len(mask)
                self.mean_vecs.append(mean)



prep = HSI_processing()
prep.load_data("data/M3G20081129T171431_V01_RFL.HDR")
# prep.visualize_raw_data("data/M3G20081118T222604_V01_RFL.HDR")
prep.KMeans("data/M3G20081129T171431_V01_RFL.HDR")
prep.get_cluster_mean()
print(prep.mean_vecs)
# print(type(prep.ClusteredPixels))
#
# data = pd.read_csv('output_list.txt', sep=" ", header=None)
# data.columns = ["a", "b", "c", "etc."]