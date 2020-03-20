from mainModel import CNN
import numpy as np
import matplotlib.pyplot as plt

# save np.load
np_load_old = np.load

# modify the default parameters of np.load
np.load = lambda *a,**k: np_load_old(*a, allow_pickle=True, **k)

# loading the data
x = np.load('xData.npy')
y = np.load('yData.npy')

# restore np.load for future normal usage
np.load = np_load_old

#maikng the model
model = CNN(0.005, 0.000001, None, None)
noOfImagesToFeed = 10

Losses = []
imageNumber = []

for i in range(noOfImagesToFeed):
	model.SetImage(x[0], y[0])
	model.ForwardPass()
	print('Loss = ' + str(model.Loss))
	Losses.append(model.Loss)
	imageNumber.append(i)
	model.BackwardPass()
	model.GradientUpdate()
	print(str(i+1) + '/' + str(noOfImagesToFeed) + ' images completed.')

weights = [model.filtersLayer1, model.filtersLayer2, model.weightsFCLayer1, model.weightsFCLayer2]
biases = [model.biasesLayer1, model.biasesLayer2, model.biasesFCLayer1, model.biasesFCLayer2]

np.save('weights.npy', weights)
np.save('biases.npy', biases)

plt.plot(imageNumber, Losses)
plt.xlabel('Number of Images')
plt.ylabel('Loss')
plt.show()



