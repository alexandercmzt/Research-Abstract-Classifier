import tensorflow as tf
import numpy as np

class Model(object):
	"""RNN Model with ICA preprocessing"""
	def __init__(self, input_dim, output_dim, num_layers, num_units, trainable, batch_size=1, lstm=False, learning_rate=0.1, grad_clip=5):
		self.input_data = tf.placeholder(tf.float32, [batch_size, input_dim])
		self.label_data = tf.placeholder(tf.int32, [batch_size])

		if lstm:
			self.cell = tf.nn.rnn_cell.BasicLSTMCell(num_units, state_is_tuple=True)
		else:
			self.cell = tf.nn.rnn_cell.GRUCell(num_units)

		if num_layers > 1:
			self.cell = tf.nn.rnn_cell.MultiRNNCell([self.cell] * num_layers, state_is_tuple=True)
		self.cell_state = self.cell.zero_state(batch_size, tf.float32)
		self.step_count = tf.Variable(0, trainable=False)

		# RNN cell update
		outputs, self.cell_state = self.cell(self.input_data, self.cell_state)

		# Map the result to a single scalar
		self.softmaxW = tf.Variable(tf.random_uniform([num_units, output_dim], minval=-1, maxval=1, dtype=tf.float32))
		self.softmaxb = tf.Variable(tf.truncated_normal([1, output_dim]), dtype=tf.float32)
		self.logits = tf.matmul(outputs, self.softmaxW) + self.softmaxb
		self.probs = tf.nn.softmax(self.logits)

		if trainable:
			self.loss = tf.nn.sparse_softmax_cross_entropy_with_logits(self.logits, self.label_data)
			trainable_vars = tf.trainable_variables()
			clipped_grads, _ = tf.clip_by_global_norm(tf.gradients(self.loss, trainable_vars), grad_clip)

			self.trainer = tf.train.AdagradOptimizer(learning_rate).apply_gradients(zip(clipped_grads, trainable_vars), self.step_count)

		self.saver = tf.train.Saver()
		
	def step(self, session, input_data, label_data=None, trainable=False):
		if trainable:
			input_feed = {self.input_data: input_data, self.label_data: label_data}
			output_var = [self.trainer, self.loss]
		else:
			input_feed = {self.input_data: input_data}
			output_var = [self.probs]
		
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