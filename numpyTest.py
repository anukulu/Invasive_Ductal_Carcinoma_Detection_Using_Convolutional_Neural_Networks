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

# arr = np.array([[1,2],[3,4]])

# arr = np.pad(arr, (1,2), mode='constant', constant_values = 0) 
# # print(arr)
# # result = [[0 0 0 0 0]
#  # 			[0 1 2 0 0]
#  # 			[0 3 4 0 0]
#  # 			[0 0 0 0 0]
#  # 			[0 0 0 0 0]]

# filters = np.random.randn(2,3,3,3)
# # print(filters)

# a = np.random.randn(3,3)
# b = np.where(a>0,a,0)
# # print(b) 

# def mult(volume):
# 	temp = np.multiply(volume, volume)
# 	return temp


# filterss = np.array([[[[-1,0,1],[-1,0,1],[-1,0,1]], [[-1,-1,-1],[0,0,0],[1,1,1]], [[0,1,1],[-1,0,1],[-1,-1,0]]],
# 					[[[-1,0,1],[-1,0,1],[-1,0,1]], [[-1,-1,-1],[0,0,0],[1,1,1]], [[0,1,1],[-1,0,1],[-1,-1,0]]],
# 					[[[-1,0,1],[-1,0,1],[-1,0,1]], [[-1,-1,-1],[0,0,0],[1,1,1]], [[0,1,1],[-1,0,1],[-1,-1,0]]]])

# filterss = mult(filterss)
# # print(filterss)

# filterss = np.where(filterss < 0, 0, filterss)
# # print(filterss)
# flattened = filterss.flatten()
# # print(len(filterss))

# arr = np.array([1,2,3,4,4,5])
# arr1 = np.reshape(arr, (6,1))



# a = mult(np.array([ [[1,2],[3,4],[5,6]], [[2,3],[5,6],[7,5]] ]))


# # print(a)

# b = np.random.randint(0,20, size=(3,3,3))
# print(b)

# c = np.flip(b, axis=(1,2))

# print(c)
# add = np.sum(filterss)
# print(add)

# b = np.random.randn(2,2,3)

# print(b)

# testImage = Image.open("test_image.png")
# img = np.array(testImage)
# plt.imshow(img[0,:,:])
# plt.show()


# g = np.array([[0],[1]])

# h = -1 / g

# print(h)

# print('\n')

# arm = np.array([[1,0],[0,0]])
# print(np.where(arm > 0))


# arm2 = np.array([[1,2,3], [4,5,6], [7,8,9]])
# ee = np.flip(arm2, (0,1))
# print(ee)

a = np.array([[1,2,3,4,5],[6,7,8,9,10],[11,12,13,14,15],[16,17,18,19,20],[21,22,23,24,25]])
column = np.zeros((1, a.shape[0]))
for i in range(2* a.shape[1] - 1):
	if (i % 2 != 0):
		a = np.hstack((a[:,:i], np.transpose(column), a[:,i:]))
a = np.transpose(a)
row = np.zeros((1, a.shape[0]))
for i in range(2* a.shape[1] - 1):
	if(i % 2 != 0):
		a = np.hstack((a[:,:i], np.transpose(row), a[:,i:]))
a = np.transpose(a)
print(a)