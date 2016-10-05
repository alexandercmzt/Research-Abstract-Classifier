from sklearn import svm
from sklearn import linear_model
from sklearn.neural_network import MLPClassifier
from sklearn.externals import joblib
from sklearn.model_selection import cross_val_score
import collections
import numpy as np
import csv

X_train = joblib.load('saves/data/train_in.csv_feature_vectors.pkl')
X_test = joblib.load('saves/data/test_in.csv_feature_vectors.pkl')
y_train = joblib.load('saves/data/train_out.csv_y_vector.pkl')


def accuracy(gold, predict):
    assert len(gold) == len(predict)
    corr = 0
    for i in xrange(len(gold)):
        if gold[i] == predict[i]:
            corr += 1
    acc = float(corr) / len(gold)
    print 'Accuracy %d / %d = %.4f' % (corr, len(gold), acc)

print collections.Counter(y_train)
# clf = linear_model.LogisticRegression()
# scores = cross_val_score(clf, X_train, y_train, cv=3)
# print("LOGREG Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))
# clf = svm.SVC()
# scores = cross_val_score(clf, X_train, y_train, cv=3)
# print("SVM Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))
# clf = MLPClassifier(solver='lbgfs', alpha=1e-5,hidden_layer_sizes=(50,20,5), random_state=1)
# scores = cross_val_score(clf, X_train, y_train, cv=3)
# print("MLP Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))


X_train = X_train.tolist()
y_train = y_train.tolist()
X = []
y = []
for i in xrange(len(y_train)):
	if y_train[i] != 'category':
		X.append(X_train[i])
		y.append(y_train[i])
X_train = np.array(X)
y_train = np.array(y)

# partition = -1*int(len(y_train)*(0.995))
# clf = linear_model.LogisticRegression()
# clf.fit(X_train[:partition], y_train[:partition])
# predictions = clf.predict(X_train[partition:])
# gold = y_train[partition:]
# accuracy(gold,predictions)


clf = linear_model.LogisticRegression()
clf.fit(X_train,y_train)
predictions = clf.predict(X_test)
output = [['id', 'category']]
for i in xrange(len(predictions)):
	output.append([i, predictions[i]])

with open("output1.csv", "wb") as f:
    writer = csv.writer(f)
    writer.writerows(output)


