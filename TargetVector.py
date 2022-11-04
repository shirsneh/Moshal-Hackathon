import numpy as np

target_filename = 'data/s07_M3t_Olivine_GDS70.a_Fo89_165um_NIC4bbb_RREF.txt'

with open(target_filename) as tf:
    target_list = tf.readlines()[93:]
    target_arr = []

    for j in target_list:
        j = j.replace("\n", "")

    for i in range(0, len(target_list) - 1):
        if 2 * (1 + i) <= len(target_list):
            target_arr[i] = (int(target_list[i]) + int(target_list[i+1])) / 2
        target_arr[i] = target_arr[i]*1000
    print(target_arr)

