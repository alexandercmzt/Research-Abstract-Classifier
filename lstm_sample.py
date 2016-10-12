from __future__ import division


from write_csv import write_csv
from test_run import prepare_features

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

_,_, X_test = prepare_features('800',args.unig)
print "finished getting X_test"

batched_input = np.split(X_test, 4)

model = lstm.Model(input_dim=800+args.unig, output_dim=4, num_layers=args.layers, num_units=args.units, trainable=False, batch_size=int(X_test.shape[0] / 4), lstm=args.lstm)

with tf.Session() as sess:
	sess.run(tf.initialize_all_variables())
	model.load(sess, args.load)
	print "loaded model"
	pred_list = []
	for b in batched_input:
		pred = model.step(sess, b)
		pred = np.argmax(pred, axis=1)
		pred_list.append(pred)
	print "about to write to csv"
	write_csv(args.out, np.concatenate(pred_list, axis=0))