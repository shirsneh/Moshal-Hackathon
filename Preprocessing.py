import glob

import matplotlib.pyplot as plt
import numpy as np
import scipy
from matplotlib.backends.backend_pdf import PdfPages
from sklearn.cluster import KMeans
from spectral import *
class HSI_processing():
    _image = None
    _data = None
    shape = None
    num_samples = None
    num_lines = None
    num_bands = None

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
            HSlayers = [5, 30, 50, 82]
            ChosenLayer = HSlayers[0]
            for ChosenLayer in HSlayers:
                fig, ax = plt.subplots(figsize=(10, 7))
                plt.title(
                    f"ImageName:  \n ImageSize:{self.num_lines}x{self.num_samples} - {self.num_bands} Layers \n Layer:{ChosenLayer}")
                ImageNP = np.asarray(_data[:, :, ChosenLayer])  # take the 2D matrix of the chosen layer
                ax.set(xlabel='Xaxis', ylabel='Yaxis')
                plt.imshow(ImageNP)  # show it
                plt.show()
                print(f"Finish image process  | size: {self.shape} | Layer:{ChosenLayer}")

    def KMeans(self , header_path):
        _image = envi.open(header_path)
        _data = _image.load()
        DataSet = []  # generate data set for each pixel, a set of fixed X,Y - take all the layers combined as a vector
        for x in range(0, self.num_lines):
            for y in range(0, self.num_samples):
                DataSet.append(_data.read_pixel(x,y))  # for each X,Y - take all the laywers as vector
        Segmentations = [2, 3, 4, 5]
        for Clusters in Segmentations:
            print(f"Training {Clusters} Clusters | size: {self.shape} | Layer:")
            kmeans = KMeans(n_clusters=Clusters)  # create kmean model with number of clusters
            kmeans.fit(DataSet)  # train the model on the data
            # Check for each x,y position the cluster it belongs
            ClusteredPixel = np.zeros((self.num_lines, self.num_samples), dtype=int)
            for x in range(0, self.num_lines):
                for y in range(0, self.num_samples):
                    ClusteredPixel[x, y] = kmeans.predict(_data[x, y, :].reshape(1, -1).astype(float))
            fig, ax = plt.subplots(figsize=(10, 7))
            plt.title(
                f"ImageName:  \n ImageSize:{self.num_lines}x{self.num_samples} - {self.num_bands} Layers \n Layer:"
                f"\n AI Algorithm for segmentation \n Image splited to {Clusters} different segmentation")
            plt.imshow(ClusteredPixel)
            plt.show()
            ax.set(xlabel='Xaxis', ylabel='Yaxis')
prep = HSI_processing()
prep.load_data("data/M3G20081118T222604_V01_RFL.HDR")
# prep.visualize_raw_data("data/M3G20081118T222604_V01_RFL.HDR")
prep.KMeans("data/M3G20081118T222604_V01_RFL.HDR")
