from mainModel import CNN
import numpy as np

# save np.load
np_load_old = np.load

# modify the default parameters of np.load
np.load = lambda *a,**k: np_load_old(*a, allow_pickle=True, **k)

# loading the data
x = np.load('testingData.npy')
w = np.load('weights.npy')
b = np.load('biases.npy')

# restore np.load for future normal usage
np.load = np_load_old

model = CNN(0.005, 0.000001, w, b)
noOfCorrectPrdeictions = 0
noOfImagesToFeed = 5


for i in range(noOfImagesToFeed):
	model.SetImage(x[i][0], x[i][1])
	model.ForwardPass()
	temp = model.softmaxFCLayer2
	a = np.zeros(np.ravel(temp).shape)
	if(temp[0][0] > temp[1][0]):
		a[0] = 1
	else:
		a[1] = 1
	if(a[0] == model.category[0] and a[1] == model.category[1]):
		noOfCorrectPrdeictions = noOfCorrectPrdeictions + 1
	print('Image number ' + str(i) + ' completed')
	print('accuracy = ' + str(noOfCorrectPrdeictions/ (i+1)))