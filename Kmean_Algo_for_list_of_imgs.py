#load all the packages needed
import scipy.io,glob,matplotlib
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.cluster import KMeans


matplotlib.use('Agg') #Use this line if you dont want to open the plot when the code is run
HSimageList=glob.glob('/Users/dvirkenig/PycharmProjects/MLPlayground2022/hs/*') #load the list of all the img files - direct it to the folder
print(f"Starting to process {len(HSimageList)} files")
for HSimageAdress in HSimageList:
    # HSimageAdress = HSimageList[0]
    HSimageMatFile = scipy.io.loadmat(HSimageAdress) #conver .mat file to array in python
    ImageName = list(HSimageMatFile.keys())[-1] #usually the last key of the dictonary is the name of the file
    TheHSimage=np.asarray(HSimageMatFile[ImageName]) #save the image content as a 3D array
    HSimageSize=TheHSimage.shape #get the size of the array
    print(f"Image has been loaded | name:{ImageName} | size: {HSimageSize}")
    HSlayers=[1,20,50,60] #chose which layers you would like to save
    HSPDFReport = PdfPages(f"pdf/HyperSpectralicImage_{ImageName}_{HSimageSize[-1]}L.pdf") #open PDF file so we can save img inside
    ChosenLayer=HSlayers[0]
    for ChosenLayer in HSlayers:
        fig, ax = plt.subplots(figsize=(10, 7))
        plt.title(f"ImageName: {ImageName} \n ImageSize:{HSimageSize[0]}x{HSimageSize[1]} - {HSimageSize[2]} Layers \n Layer:{ChosenLayer}")
        ImageNP = np.asarray(TheHSimage[:,:,ChosenLayer]) #take the 2D matrix of the chosen layer
        plt.imshow(ImageNP) #show it
        ax.set(xlabel='Xaxis', ylabel='Yaxis')
        HSPDFReport.savefig(fig) #Save it in the PDF file
        print(f"Finish image process | name:{ImageName} | size: {HSimageSize} | Layer:{ChosenLayer}")

    #Pre Processing - prepare the data set in pixel driven batch. for each pixel take all the layers
    DataSet=[] #generate data set for each pixel, a set of fixed X,Y - take all the layers combined as a vector
    for x in range(0,HSimageSize[0]):
        for y in range(0,HSimageSize[1]):
            DataSet.append(TheHSimage[x,y,:]) #for each X,Y - take all the laywers as vector
    DataSet = np.array(DataSet) #convert it to numpy array
    #Run Kmean algorihem to get the centers for each pixel
    Segmentations=[2,3,4,5]
    for Clusters in Segmentations:
        # Clusters=Segmentations[0]
        print(f"Training {Clusters} Clusters | name:{ImageName} | size: {HSimageSize} | Layer:{ChosenLayer}")
        kmeans = KMeans(n_clusters=Clusters) #create kmean model with number of clusters
        kmeans.fit(DataSet) #train the model on the data
        #Check for each x,y position the cluster it belongs
        ClusteredPixel=np.zeros((HSimageSize[0],HSimageSize[1]), dtype=int)
        for x in range(0,HSimageSize[0]):
            for y in range(0,HSimageSize[1]):
                ClusteredPixel[x,y]=kmeans.predict(TheHSimage[x,y,:].reshape(1, -1))
        fig, ax = plt.subplots(figsize=(10, 7))
        plt.title(f"ImageName: {ImageName} \n ImageSize:{HSimageSize[0]}x{HSimageSize[1]} - {HSimageSize[2]} Layers \n Layer:{ChosenLayer}"
                  f"\n AI Algorithm for segmentation \n Image splited to {Clusters} different segmentation")
        plt.imshow(ClusteredPixel)
        ax.set(xlabel='Xaxis', ylabel='Yaxis')
        HSPDFReport.savefig(fig)

    print(f"Finish to generate PDF | name:{ImageName} | size: {HSimageSize}")
    HSPDFReport.close()
print('Finish the code')


