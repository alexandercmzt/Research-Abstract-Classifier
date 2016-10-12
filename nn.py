import tensorflow as tf
import numpy as np

class Model():
	def __init__(self, input_dim, output_dim, hidden_dim, trainable, learning_rate=0.1, grad_clip = 5.0):
		self.input_dim = input_dim
		self.output_dim = output_dim

		self.num_layers = len(hidden_dim)
		self.hidden_dim = hidden_dim

		self.learning_rate = learning_rate

		self.W = []
		self.b = []
		prev_dim = input_dim
		for h in xrange(self.num_layers):
			self.W.append(tf.Variable(tf.truncated_normal([prev_dim, hidden_dim[h]]), name='W_{}'.format(h), trainable=True))
			self.b.append(tf.Variable(tf.truncated_normal([hidden_dim[h]]), name='b_{}'.format(h), trainable=True))
			prev_dim = hidden_dim[h]

		self.W.append(tf.Variable(tf.truncated_normal([hidden_dim[-1], output_dim]), name='W_out', trainable=True))
		self.b.append(tf.Variable(tf.truncated_normal([output_dim]), name='b_out', trainable=True))

		self.input_data = tf.placeholder(tf.float32, [None, self.input_dim])
		self.label_data = tf.placeholder(tf.int32, [None])

		self.stepcount = tf.Variable(0, trainable=False)

		self.hidden = []
		prev_layer = self.input_data
		for h in xrange(self.num_layers + 1):
			self.hidden.append(tf.sigmoid(tf.matmul(prev_layer, self.W[h]) + self.b[h]))
			prev_layer = self.hidden[h]

		self.logits = tf.nn.softmax(prev_layer)

		if trainable:
			self.loss = tf.nn.sparse_softmax_cross_entropy_with_logits(prev_layer, self.label_data, name='loss')
			trainable_vars = tf.trainable_variables()
			self.gradients = tf.gradients(self.loss, trainable_vars)
			if grad_clip > 0.0:
				clipped_grads, _ = tf.clip_by_global_norm(self.gradients, grad_clip)
				self.trainer = tf.train.AdagradOptimizer(self.learning_rate).apply_gradients(zip(clipped_grads, trainable_vars), self.stepcount)
			else:
				self.trainer = tf.train.AdagradOptimizer(self.learning_rate).apply_gradients(zip(self.gradients, trainable_vars), self.stepcount)

		self.saver = tf.train.Saver()

	def step(self, session, input_data, label_data=None, trainable=False):
		if trainable:
			input_feed = {self.input_data: input_data, self.label_data: label_data}
			output_var = [self.trainer, self.loss]
		else:
			input_feed = {self.input_data: input_data}
			output_var = [self.logits]
		output = session.run(output_var, feed_dict=input_feed)
		
		if trainable:
			return output[1]
		else:
			return output[0]

	def save(self, session, file):
		save_path = self.saver.save(session, file)
		print "Model saved at {}".format(save_path)

	def load(self, session, file):
		self.saver.restore(session, file)
		print "Model loaded from {}".format(file)