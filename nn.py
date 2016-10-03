import tensorflow as tf
import numpy as np

class Model():
	def __init__(self, input_dim, output_dim, hidden_dim, trainable, learning_rate, grad_clip = 0.0):
		self.input_dim = 400
		self.output_dim = 4

		self.num_layers = len(hidden_dim)
		self.hidden_dim = hidden_dim

		self.learning_rate = learning_rate

		self.W = []
		self.b = []
		prev_dim = input_dim
		for h in xrange(num_layers):
			self.W.append(tf.Variable(tf.truncated_normal([prev_dim, hidden_dim[h]])), name='W_{}'.format(h))
			self.b.append(tf.Variable(tf.truncated_normal([hidden_dim[h]])), name='b_{}'.format(h))
			prev_dim = hidden_dim[h]

		self.W.append(tf.Variable(tf.truncated_normal([hidden_dim[-1], output_dim])), name='W_out')
		self.b.append(tf.Variable(tf.truncated_normal([output_dim])), name='b_out')	

		self.input_data = tf.placeholder(tf.float32, [None, self.input_dim])
		self.label_data = tf.placeholder(tf.float32, [None, self.label_data])

		self.stepcount = tf.Variable(0, trainable=false)

		self.hidden = []
		prev_layer = input_data
		for h in xrange(num_layers + 1):
			self.hidden.append(tf.sigmoid(tf.matmul(prev_layer, self.W[h]) + self.b[h]))
			prev_layer = self.hidden[h]

		self.logits = tf.softmax(prev_layer, dim=1)

		if trainable:
			self.loss = tf.nn.softmax_cross_entropy_with_logits(self.logits, self.label_data, dim=1, name='loss')
			trainable_vars = tf.trainable_variables()
			self.gradients = tf.gradient(self.loss, trainable_vars)
			if grad_clip > 0.0:
				clipped_grads, _ = tf.clip_by_global_norm(self.gradients, arg.gradient_clip)
				self.trainer = tf.train.AdamOptimizer(self.learning_rate).apply_gradients(zip(clipped_grads, trainable_vars), self.stepcount)
			else:
				self.trainer = tf.train.AdamOptimizer(self.learning_rate).apply_gradients(zip(self.gradients, trainable_vars), self.stepcount)

		self.saver = tf.train.Saver()

	def step(self, session, input_data, label_data=None, trainable=False):
		if trainable:
			input_feed = {self.input_data: input_data, self.label_data: label_data}
			output_var = [self.trainer, self.loss]
		else:
			input_feed = {self.input_data: input_data}
			output_var = [self.self.logits]
		output = session.run(output_var, feed_dict=input_feed)
		
		if trainable:
			return output_var[1]
		else:
			return output_var[0]

	def save(self, session, file):
		save_path = self.saver.save(session, file)
		print "Model saved at {}".format(save_path)

	def load(self, session, file):
		self.saver.restore(session, file)
		print "Model loaded from {}".format(file)