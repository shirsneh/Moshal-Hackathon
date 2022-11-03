import matplotlib.pyplot as plt
from spectral import *
# Parameters.
# input_filename = "M3G20081118T222604_V01_RFL.IMG"
# shape = (304, 1182, 85)  # matrix size
# dtype = np.dtype('>u2')  # big-endian unsigned integer (16bit)
# output_filename = "M3G20081118T222604_V01_RFL.PNG"
#
# # Reading.
# fid = open(input_filename, 'rb')
# data = np.fromfile(fid, dtype)
# image = data.reshape(int(61085760 / 2), 2)
#
# # Display.
# plt.imshow(data, cmap="gray")
# # plt.savefig(output_filename)
# plt.show()
image = envi.open('M3G20081118T222604_V01_RFL.HDR')
data = image.load()
# pc = principal_components(image)
# v = imshow(pc.cov)
# pc_0999 = pc.reduce(fraction=0.999)
# len(pc_0999.eigenvalues)
# img_pc = pc_0999.transform(image)
# v = imshow(img_pc[:,:,:3], stretch_all=True)
# plt.show()
(m, c) = kmeans(data[: , : , 4], nclusters=3, max_iterations=20)
plt.imshow(m)
plt.show()
