import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import png

class CNN:
	def __init__(self, learningRate):
		self.learningRate = learningRate

	def Convolution2D(self, fltr, layer, stride, pad):

		outputHeight = int(((layer.shape[0] - fltr.shape[0] + 2 * pad) / stride) + 1) 
		outputWidth = int(((layer.shape[1] - fltr.shape[1] + 2 * pad)/ stride) + 1)

		convolutedImage = np.zeros((outputHeight, outputWidth))

		for i in range(int(layer.shape[0] - 2 * pad)):
			for j in range(int(layer.shape[1] - 2 * pad)):

				convolutedImage[i,j] = np.sum(fltr * layer[i:(i + fltr.shape[0]) , j: (j + fltr.shape[1])])

		return convolutedImage

	def Convolution3D(self, originalImage, fltr, pad, stride):

		# new image after padding each of the layers
		image = np.zeros((originalImage.shape[0] + 2 * pad, originalImage.shape[1] + 2 * pad, originalImage.shape[2]))

		for i in range(originalImage.shape[2]):
			# padding each layer and assigning to the new image
			image[:,:,i] = np.pad(originalImage[:,:,i], (pad,pad), mode='constant', constant_values = 0)

		outputHeight = int(((originalImage.shape[0] - fltr.shape[1] + 2 * pad) / stride) + 1) 
		outputWidth = int(((originalImage.shape[1] - fltr.shape[2] + 2 * pad)/ stride) + 1)
		outputDepth = fltr.shape[0]

		convolutedVolume = np.zeros((outputHeight, outputWidth, outputDepth))

		for x in range(fltr.shape[0]):

			convolutedImage = np.zeros((outputHeight, outputWidth))

			for y in range(fltr.shape[3]):

				convolutedImage += self.Convolution2D(fltr[x, :, :, y], originalImage[:,:,y], stride, pad)

			convolutedVolume[:,:,x] = convolutedImage

		return convolutedVolume


cnn = CNN(0.01)

testImage = Image.open("test_image.png")
img = np.array(testImage)

filters = np.random.randn(3,3,3,3)

output = cnn.Convolution3D(img, filters, 1, 1)
print(output.shape)

firstSlice = output[:,:,0]

# imgplot = plt.imshow(firstSlice)
# plt.show()

plot = plt.imshow(filters[0])
plt.show()




