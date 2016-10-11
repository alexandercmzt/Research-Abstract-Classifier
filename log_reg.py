import numpy as np

class Model():
	def __init__(self, input_dim, learning_rate=0.1):
		self.input_dim = input_dim
		self.learning_rate = learning_rate
		self.w = np.random.normal(size=[input_dim+1,1])

	def forward(self, x):
		"""Allow for batch processing, assume row-major"""
		if x.shape[1] == self.w.shape[0] - 1:
			x = np.concatenate((np.ones([x.shape[0], 1]), x), axis=1)
		return np.round(self.sigmoid(np.dot(x, self.w)))

	def sigmoid(self, x):
		return 1 / (1 + np.exp(-x))

	def gradient(self, x, y):
		# print x.shape, (y-self.forward(x)).shape, self.w.shape
		return np.dot((y - self.forward(x)), x)

	def step(self, x, y, e=float("inf")):
		"""By default, this function performs one update step
			error epsilon can be changed to perform multi updates at once"""
		x = np.concatenate((np.ones([x.shape[0], 1]), x), axis=1)
		self.w += self.learning_rate * np.sum(self.gradient(x,y), axis=0).T.reshape(self.input_dim + 1, 1)

	def save(self, filename):
		np.save(filename, self.w, allow_pickle=False)

	def load(self, filename):
		np.load(filename, self.w, allow_pickle=False)
