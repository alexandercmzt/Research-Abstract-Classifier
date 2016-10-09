import csv
import numpy as np
import sys
from sklearn.externals import joblib

if len(sys.argv) != 2:
	print "Please use the path to the .csv file as the first command line argument."

filename = sys.argv[1]

with open(filename, 'rU') as f:
	reader = csv.reader(f)
	data = list(list(elem) for elem in csv.reader(f, delimiter=','))
	#remove headers from table
	data = data[1:]

for i in xrange(len(data)):
	data[i] = data[i][1]

print np.array(data)
print np.array(data).shape
joblib.dump(np.array(data), 'saves/' + filename +'_y_vector.pkl')
print "y vector saved."