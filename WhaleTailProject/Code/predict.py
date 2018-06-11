import deepneuralnet as net
import numpy as np
import sklearn.metrics as metrics
from sklearn.metrics import classification_report,accuracy_score
from tflearn.data_utils import image_preloader
model = net.model
path_to_model = './ZtrainedNet/final-model.tfl'
model.load(path_to_model)
X, Y = image_preloader(target_path='./validate', image_shape=(100,100), mode='folder',
 grayscale=False, categorical_labels=True, normalize=True)
X = np.reshape(X, (-1, 100, 100, 3))

y_pred = []
y_test = []
for i in range(0, len(X)):
	iimage = X[i]
	icateg = Y[i]
	result = model.predict([iimage])[0]
	prediction = result.tolist().index(max(result))
	reality = icateg.tolist().index(max(icateg))
	y_pred.append(prediction)
	y_test.append(reality)
	if prediction == reality:
		print("image %d CORRECT " % i, end='')
	else:
		print("image %d WRONG " % i, end='')
	print(result)
	
print("Accuracy: "+str(accuracy_score(y_test, y_pred)))
print('\n')
print(classification_report(y_test, y_pred))