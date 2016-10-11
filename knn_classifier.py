from sklearn.externals import joblib
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import normalize
from sklearn.neighbors import KNeighborsClassifier
from scipy.stats import mode
import collections
import numpy as np
import csv
import math
import time

X_train = joblib.load('saves/data/train_in.csv_feature_vectors_800.pkl')#.tolist()
X_test = joblib.load('saves/data/test_in.csv_feature_vectors_800.pkl')#.tolist()
y_train = joblib.load('saves/data/train_out.csv_y_vector.pkl')#.tolist()

X_test_split = X_train[:10000]
y_test_split = y_train[:10000]

X_train_split = X_train[10000:]
y_train_split = y_train[10000:]


def accuracy(gold, predict):
    assert len(gold) == len(predict)
    corr = 0
    for i in xrange(len(gold)):
        if gold[i] == predict[i]:
            corr += 1
    acc = float(corr) / len(gold)
    print 'Accuracy %d / %d = %.4f' % (corr, len(gold), acc)
    return acc

class knn_classifier(object):
	""" Class that implements k nearest neighbor algorithm """
	def __init__(self, numNeighbors):
		self.numNeighbors = numNeighbors
		self.label_to_class = {0 : 'stat', 1 : 'math', 2 : 'physics', 3 : 'cs', 4: 'category'}
		self.class_to_label = {'stat' : 0, 'math' : 1, 'physics' : 2, 'cs' : 3, 'category': 4}

	def predict(self, inputVector):
		""" Finds self.numNeighbors nearest neighbors to the inputVector and returns predicted class """
		# inputVector = np.asarray(inputVector)
		start = time.time()
		times = []
		distances = []
		idx = 0

		# slow code, due to using python iterator
		"""
		for vector in X_train: # calculate distance between inputVector and every vector in the dataset
			dist = np.linalg.norm(np.asarray(vector) - inputVector) # euclidian distance in np
			distances.append((dist, idx))
			idx += 1
		"""

		# matrix subtraction that calculates all distances in a single line using numPy operations
		distVect = np.linalg.norm(X_train - inputVector, axis=1)

		for distValue in distVect:
			distances.append((distValue, idx))
			idx += 1
		
		distances = sorted(distances, key = lambda distance: distance[0]) # sort vectors in dataset by distance

		res = distances[:self.numNeighbors]
		neighbs = []
		for aResult in res:
			neighbs.append(self.class_to_label[y_train[aResult[1]]])
			times.append(time.time() - start)

		avgTime = sum(times)/len(times)
		print 'Mean prediction time is %f seconds' % avgTime
		return self.label_to_class[mode(np.asarray(neighbs))[0][0]]

resultAccs = []

# Find metrics for neighbor values of 3 to 8
for i in range(3, 9):
	# our classifier
	tristan_knn = knn_classifier(i)
	predictions = []
	counter = 0
	for trainingExample in X_test_split:
		print 'Predicting example: %s/%i' % (counter+1, len(X_test_split))
		predictions.append(tristan_knn.predict(trainingExample))
		counter += 1

	print 'our knn model:'
	our = accuracy(y_test_split, predictions)

	# sklearn comparison
	neigh = KNeighborsClassifier(n_neighbors=i)
	neigh.fit(X_train_split, y_train_split)

	predictions = []
	counter = 0
	for trainingExample in X_test_split:
		print 'Predicting example: %s/%i' % (counter+1, len(X_test_split))
		predictions.append(neigh.predict([X_train[1]])[0])
		counter += 1

	print 'sklearn model:'
	skl = accuracy(y_test_split, predictions)

	resultAccs.append((our, skl))

for res in resultAccs:
	print res
