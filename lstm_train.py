from __future__ import division
import argparse

parser = argparse.ArgumentParser(description="LSTM/GRU trainer")
parser.add_argument('--load', type=str, default=None)
parser.add_argument('--save', type=str, default=None)
parser.add_argument('--unig', type=int, default=20000)
parser.add_argument('--units', type=int, default=400)
parser.add_argument('--layers', type=int, default=2)
parser.add_argument('--batch_size', type=int, default=50)
parser.add_argument('--lstm', dest='lstm', action='store_true')
parser.add_argument('--gru', dest='lstm', action='store_false')
parser.set_defaults(lstm=False)
args = parser.parse_args()


import tensorflow as tf
import numpy as np
import lstm
from test_run import prepare_features

from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

X_train,y_train,_ = prepare_features('800',args.unig)

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

X_train, X_test, y_train, y_test = train_test_split(X_train, y_train, test_size=0.1, random_state=0)

model = lstm.Model(input_dim=800+args.unig, output_dim=4, num_layers=args.layers, num_units=args.units, trainable=True, batch_size=args.batch_size, lstm=args.lstm)

to_remove = X_train.shape[0] % args.batch_size
X_train = np.delete(X_train, range(X_train.shape[0]-1-to_remove,X_train.shape[0]-1), axis=0)
y_train = np.delete(y_train, range(X_train.shape[0]-1-to_remove,X_train.shape[0]-1), axis=0)
num_bin = X_train.shape[0] / args.batch_size
batched_input = np.split(X_train, num_bin)
batched_label = np.split(y_train, num_bin)

to_remove = X_test.shape[0] % args.batch_size
X_test = np.delete(X_test, range(X_test.shape[0]-1-to_remove,X_test.shape[0]-1), axis=0)
y_test = np.delete(y_test, range(X_test.shape[0]-1-to_remove,X_test.shape[0]-1), axis=0)
num_bin = X_test.shape[0] / args.batch_size
test_batched_input = np.split(X_test, num_bin)
test_batched_label = np.split(y_test, num_bin)

with tf.Session() as sess:
	sess.run(tf.initialize_all_variables())
	if args.load is not None:
		model.load(sess, args.load)
	for i in xrange(10000):
		total_loss = 0.0
		for b in xrange(len(batched_input)):
			loss = model.step(sess, batched_input[b], batched_label[b], trainable=True)
			loss += np.sum(loss) / len(loss)
		print "Iteration {} with loss {}".format(i, total_loss / num_bin)

		#if i % 10 == 0:
		accuracy = []
		for b in xrange(len(test_batched_input)):
			pred = model.step(sess, test_batched_input[b])
			pred = np.argmax(pred, axis=1)
			acc = accuracy_score(test_batched_label[b], pred)
			accuracy.append(acc)
		print '### Iteration {} with ACCURACY: {} ###'.format(i, np.sum(accuracy)/len(accuracy))

		if args.save is not None:
			model.save(sess, args.save)