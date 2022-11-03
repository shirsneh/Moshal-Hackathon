import scipy.io as sp
import re
import matplotlib.pyplot as plt
class preprocessing():
    _filename_regex = r"data/(.*).mat"
    _data = []
    def get_single_HSI(self, filepath):
        filename = re.search(self.filename_regex, filepath).group(1)
        HS_dict = sp.loadmat(filepath)
        HSI = HS_dict[filename]
        self._data.append(HSI)


