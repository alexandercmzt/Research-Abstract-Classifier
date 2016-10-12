import log_reg
import numpy as np
from sklearn import datasets
import matplotlib.pyplot as plt

mu_vec1 = np.array([0,0])
cov_mat1 = np.array([[2,0],[0,2]])
x1_samples = np.random.multivariate_normal(mu_vec1, cov_mat1, 100)
mu_vec1 = mu_vec1.reshape(1,2).T # to 1-col vector

mu_vec2 = np.array([1,2])
cov_mat2 = np.array([[1,0],[0,1]])
x2_samples = np.random.multivariate_normal(mu_vec2, cov_mat2, 100)
mu_vec2 = mu_vec2.reshape(1,2).T

X = np.concatenate((x1_samples,x2_samples), axis = 0)
Y = np.array([0]*100 + [1]*100)

model = log_reg.Model(2, 0.1)

for i in range(10):
	model.step(X,Y)

# Following decision boundary plotting code is a modified version of the first answer in: http://stackoverflow.com/questions/19054923/plot-decision-boundary-matplotlib
# X - some data in 2dimensional np.array
x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
h=0.01
xx, yy = np.meshgrid(np.arange(x_min, x_max, h),np.arange(y_min, y_max, h))
# here "model" is your model's prediction (classification) function
Z = model.forward(np.c_[xx.ravel(), yy.ravel()])
# Put the result into a color plot
Z = Z.reshape(xx.shape)
plt.contourf(xx, yy, Z, cmap=plt.cm.Paired)
plt.axis('off')
# # Plot also the training points
# plt.scatter(X[:, 0], X[:, 1], c=Y, cmap=plt.cm.Paired)
plt.scatter(x1_samples[:,0],x1_samples[:,1], c= 'red', marker='o')
plt.scatter(x2_samples[:,0],x2_samples[:,1], c= 'green', marker='o')
plt.show()