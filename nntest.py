import tensorflow as tf
import numpy as np
import nn

from sklearn.externals import joblib
print "loaded"

X_train = joblib.load('saves/data/train_in.csv_feature_vectors.pkl')
X_test = joblib.load('saves/data/test_in.csv_feature_vectors.pkl')
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
temp = np.zeros((y_train.shape[0], 4))
temp[np.arange(y_train.shape[0]), y_train] = 1
y_train = temp

model = nn.Model(400, 4, [500,500], True, 0.1)

with tf.Session() as sess:
	sess.run(tf.initialize_all_variables())
	print len(sess.run(tf.trainable_variables()))
	# for i in xrange(100):
	# 	loss = model.step(sess, X_train, y_train, trainable=True)
	# 	print "Iteration {}  with loss {}".format(sess.run(model.stepcount), loss)