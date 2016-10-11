from __future__ import division


from write_csv import write_csv

import argparse

parser = argparse.ArgumentParser(description="LSTM/GRU sampler")
parser.add_argument('--load', type=str)
parser.add_argument('--unig', type=int, default=20000)
parser.add_argument('--units', type=int, default=400)
parser.add_argument('--layers', type=int, default=2)
parser.add_argument('--lstm', dest='lstm', action='store_true')
parser.add_argument('--gru', dest='lstm', action='store_false')
parser.set_defaults(lstm=False)
parser.add_argument('--out', type=str)
args = parser.parse_args()

import tensorflow as tf
from sklearn.externals import joblib
import numpy as np
import lstm

X_test = joblib.load('saves/data/400/test_in.csv_feature_vectors.pkl')

batched_input = np.split(X_test, X_test.shape[0])

model = lstm.Model(input_dim=800+args.unig, output_dim=4, num_layers=args.layers, num_units=args.unig, trainable=True, batch_size=1, lstm=args.lstm)

with tf.Session() as sess:
	sess.run(tf.initialize_all_variables())
	model.load(sess, args.load)

	pred_list = []
	for b in batched_input:
		pred = model.step(sess, b)
		pred = np.argmax(pred, axis=1)
		pred_list.append(pred)
	write_csv(args.out, pred_list)