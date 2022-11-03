import glob

import matplotlib.pyplot as plt
import numpy as np
import scipy
from matplotlib.backends.backend_pdf import PdfPages
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
        _data = _image.open_memmap()
        self.shape = _data.shape
        self.num_lines = _data.shape[0]
        self.num_samples = _data.shape[1]
        self.num_bands = _data.shape[2]

    def visualize_raw_data(self , header_path):
            _image = envi.open(header_path)
            _data = _image.load()
            self.shape = _data.shape
            self.num_lines = _data.shape[0]
            self.num_samples = _data.shape[1]
            self.num_bands = _data.shape[2]
            HSlayers = [1, 20, 50, 60]  # chose which layers you would like to save
            HSPDFReport = PdfPages(
                f"pdf/HyperSpectralicImage.pdf")  # open PDF file so we can save img inside
            ChosenLayer = HSlayers[0]
            for ChosenLayer in HSlayers:
                fig, ax = plt.subplots(figsize=(10, 7))
                plt.title(
                    f"ImageName:  \n ImageSize:{self.num_lines}x{self.num_samples} - {self.num_bands} Layers \n Layer:{ChosenLayer}")
                ImageNP = np.asarray(_data[:, :, ChosenLayer])  # take the 2D matrix of the chosen layer
                plt.imshow(ImageNP)  # show it
                plt.show()
                ax.set(xlabel='Xaxis', ylabel='Yaxis')
                HSPDFReport.savefig(fig)  # Save it in the PDF file
                print(f"Finish image process  | size: {self.shape} | Layer:{ChosenLayer}")

    # def KMeans(self):
    #     Segmentations = [2, 3, 4, 5]
    #     for Clusters in Segmentations:
    #         # Clusters=Segmentations[0]
    #         print(f"Training {Clusters} Clusters | name:{ImageName} | size: {HSimageSize} | Layer:{ChosenLayer}")
    #         kmeans = KMeans(n_clusters=Clusters)  # create kmean model with number of clusters
    #         kmeans.fit(DataSet)  # train the model on the data
    #         # Check for each x,y position the cluster it belongs
    #         ClusteredPixel = np.zeros((HSimageSize[0], HSimageSize[1]), dtype=int)
    #         for x in range(0, HSimageSize[0]):
    #             for y in range(0, HSimageSize[1]):
    #                 ClusteredPixel[x, y] = kmeans.predict(TheHSimage[x, y, :].reshape(1, -1))
    #         fig, ax = plt.subplots(figsize=(10, 7))
    #         plt.title(
    #             f"ImageName: {ImageName} \n ImageSize:{HSimageSize[0]}x{HSimageSize[1]} - {HSimageSize[2]} Layers \n Layer:{ChosenLayer}"
    #             f"\n AI Algorithm for segmentation \n Image splited to {Clusters} different segmentation")
    #         plt.imshow(ClusteredPixel)
    #         ax.set(xlabel='Xaxis', ylabel='Yaxis')
prep = HSI_processing()
# prep.load_data()
prep.visualize_raw_data("data/M3G20081118T222604_V01_RFL.HDR")