import numpy as np
from PIL import Image
import matplotlib.pyplot as plt 

# testImage = Image.open("test_image.png")
# image = np.array(testImage)
# print(image)
# print(image.shape)

# print(type(image))

# channel0 = image[:,:,0]
# channel1 = image[:,:,1]
# channel2 = image[:,:,2]

# print(np.array_equal(channel0, channel1))
# print(np.array_equal(channel1, channel2))

# arr = np.arange(90)
# arr = arr.reshape(9,2,5)	# arranged as rows * columns * height(no of layers)
# 							# eg : [[[1 2 3][4 5 6]]	1 2 and 3 are on separate 3 layers(height)
# 							#		[[7 8 9][10 11 12]]
# 							#		[[13 14 15][16 17 18]]]
# print(arr)
# print(arr[:,0,:])

# filter_1 = np.array([[1, 0, -1], [0, 0, 0]])
# print(filter_1.shape)

arr = np.array([[1,2],[3,4]])

arr = np.pad(arr, (1,2), mode='constant', constant_values = 0) 
print(arr)
# result = [[0 0 0 0 0]
 # 			[0 1 2 0 0]
 # 			[0 3 4 0 0]
 # 			[0 0 0 0 0]
 # 			[0 0 0 0 0]]

filters = np.random.randn(2,3,3,3)
print(filters)

a = np.random.randn(3,3)
b = np.where(a>0,a,0)
print(b) 
