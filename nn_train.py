from __future__ import division

import tensorflow as tf
import numpy as np
import nn

from sklearn.externals import joblib
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

print "loaded"

X_train = joblib.load('saves/data/train_in.csv_feature_vectors.pkl')
y_train = joblib.load('saves/data/train_out.csv_y_vector.pkl')

index=[]
for i in xrange(len(y_train)):
	if y_train[i] == 'math':
		y_train[i] = 0
	elif y_train[i] == 'stat':
		y_train[i] = 1
	elif y_train[i] == 'cs':
		y_train[i] = 2
	elif y_train[i] == 'physics':
		y_train[i] = 3
	else:
		index.append(i)

X_train = np.delete(X_train, index, axis=0)
y_train = np.delete(y_train, index, axis=0).astype(np.int)

X_train, X_test, y_train, y_test = train_test_split(X_train, y_train, test_size=0.2, random_state=0)

model = nn.Model(400, 4, [50,60,30], True, 0.1)

with tf.Session() as sess:
	sess.run(tf.initialize_all_variables())
	# model.load(sess, 'saves/model/nn551k.model')
	for i in xrange(50000):
		loss = model.step(sess, X_train, y_train, trainable=True)
		print "Iteration {} with average loss {}".format(i, np.sum(loss) / len(loss))

		if i % 100 == 0:
			pred = model.step(sess, X_test)
			pred = np.argmax(pred, axis=1)
			acc = accuracy_score(y_test, pred)
			print 'Iteration {} with accuracy: {}'.format(i, acc)

			model.save(sess, 'saves/model/nn563.model')