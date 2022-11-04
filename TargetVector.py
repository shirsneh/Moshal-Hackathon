import numpy as np

target_filename = 'data/s07_M3t_Olivine_GDS70.b_Fo89_115um_NIC4bbb_RREF.txt'

with open(target_filename) as tf:
    target_list = tf.readlines()[93:]
    target_arr = []

    for i in range(0, (len(target_list)//2 -1)):
        num1 = float(target_list[2*i].replace("\n", ""))
        num2 = float(target_list[2*i+1].replace("\n", ""))
        target_arr.append(((num1 +num2) / 2)*1000)

    print(target_arr)
    print(len(target_arr))

