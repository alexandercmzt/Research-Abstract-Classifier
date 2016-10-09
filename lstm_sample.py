from __future__ import division

import tensorflow as tf
import numpy as np
import lstm

from write_csv import write_csv

from sklearn.externals import joblib

X_test = joblib.load('saves/data/400/test_in.csv_feature_vectors.pkl')

batched_input = np.split(X_test, X_test.shape[0])

model = lstm.Model(input_dim=400, output_dim=4, num_layers=2, num_units=300, trainable=True, batch_size=1)

with tf.Session() as sess:
	sess.run(tf.initialize_all_variables())
	model.load(sess, 'saves/model/lstm2_3.model')

	pred_list = []
	for b in batched_input:
		pred = model.step(sess, b)
		pred = np.argmax(pred, axis=1)
		pred_list.append(pred)
	write_csv('testLSTM400.csv', pred_list)