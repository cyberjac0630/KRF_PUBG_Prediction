# -*- coding:utf-8 -*-
# Author:ZXW
# Time:2021/1/9
import numpy as np

demo_npy = []
demo_npy_test = []
data = np.load('data/train_no_char_2.npy')
print(data.shape)
count = 0
for i in data:
    if count < 300000:
        demo_npy.append(i)
    # if 5000 < count <= 10000:
    #     demo_npy_test.append(i)
    count = count + 1
    # print(i[:26])


result = np.array(demo_npy)
# result_test = np.array(demo_npy_test)
print(result.shape)
# print(result_test.shape)
np.save('data/train_demo_train_300000.npy', result)
# np.save('data/train_demo_test.npy', result_test)
