import scipy.io as sp
import re

import matplotlib.pyplot as plt


def get_HSI(filepath):
    filename_regex = r"data/(.*).mat"
    filename = re.search(filename_regex, filepath).group(1)
    HS_dict = sp.loadmat(filepath)
    HSI = HS_dict[filename]
    return HSI


img = get_HSI('data/salinasA.mat')
plt.imshow(img[: , : , 80])
plt.show()