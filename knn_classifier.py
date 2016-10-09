from sklearn.externals import joblib
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import normalize
from sklearn.neighbors import KNeighborsClassifier
from scipy.stats import mode
import collections
import numpy as np
import csv
import math

X_train = joblib.load('saves/data/train_in.csv_feature_vectors_800.pkl').tolist()
X_test = joblib.load('saves/data/test_in.csv_feature_vectors_800.pkl').tolist()
y_train = joblib.load('saves/data/train_out.csv_y_vector.pkl').tolist()

def accuracy(gold, predict):
    assert len(gold) == len(predict)
    corr = 0
    for i in xrange(len(gold)):
        if gold[i] == predict[i]:
            corr += 1
    acc = float(corr) / len(gold)
    print 'Accuracy %d / %d = %.4f' % (corr, len(gold), acc)

class knn_classifier(object):
	""" Class that implements k nearest neighbor algorithm """
	def __init__(self, numNeighbors):
		self.numNeighbors = numNeighbors
		self.label_to_class = {0 : 'stat', 1 : 'math', 2 : 'physics', 3 : 'cs', 4: 'category'}
		self.class_to_label = {'stat' : 0, 'math' : 1, 'physics' : 2, 'cs' : 3, 'category': 4}

	def predict(self, inputVector):
		""" Finds self.numNeighbors nearest neighbors to the inputVector and returns predicted class """
		inputVector = np.asarray(inputVector)
		distances = []
		idx = 0
		for vector in X_train: # calculate distance between inputVector and every vector in the dataset
			dist = np.linalg.norm(np.asarray(vector) - inputVector) # euclidian distance in np
			
			distances.append((dist, idx))
			idx += 1
		
		distances = sorted(distances, key = lambda distance: distance[0]) # sort vectors in dataset by distance
		res = distances[:self.numNeighbors]
		neighbs = []
		for aResult in res[1:]:
			neighbs.append(self.class_to_label[y_train[aResult[1]]])

		return self.label_to_class[mode(np.asarray(neighbs))[0][0]]

numExamples = 300

# our classifier
tristan_knn = knn_classifier(3)
predictions = []
counter = 0
for trainingExample in X_train[:numExamples]:
	print 'Predicting example: %s/%i' % (counter, numExamples)
	predictions.append(tristan_knn.predict(trainingExample))
	counter += 1

print 'accuracy of our classifier is %d.' % accuracy(y_train[:numExamples], predictions)


# sklearn comparison
neigh = KNeighborsClassifier(n_neighbors=3)
neigh.fit(X_train, y_train)

predictions = []
counter = 0
for trainingExample in X_train[:numExamples]:
	print 'Predicting example: %s/%i' % (counter, numExamples)
	predictions.append(neigh.predict([X_train[1]])[0])
	counter += 1

print 'accuracy of the sklearn classifier is %d.' % accuracy(y_train[:numExamples], predictions)




