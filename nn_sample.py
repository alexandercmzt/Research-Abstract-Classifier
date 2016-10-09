from __future__ import division

import tensorflow as tf
import numpy as np
import nn
from write_csv import write_csv

from sklearn.externals import joblib

X_test = joblib.load('saves/data/test_in.csv_feature_vectors.pkl')

model = nn.Model(400, 4, [50,60,30], False)

with tf.Session() as sess:
	sess.run(tf.initialize_all_variables())
	model.load(sess, 'saves/model/nn563.model')

	pred = model.step(sess, X_test)
	pred = np.argmax(pred, axis=1)

	write_csv('test563.csv', pred)