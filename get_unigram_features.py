import csv 
import sys, os, codecs
from os import listdir
from os.path import isfile, join
import numpy as np
import sklearn
import string
from sklearn.feature_extraction import DictVectorizer, FeatureHasher
from collections import Counter
from sklearn.externals import joblib

if len(sys.argv) != 3:
	print "Please use the path to the train.csv file as the first command line argument and the path to the test.csv file as the second command line argument"
	exit()
filename = sys.argv[1]
test_filename = sys.argv[2]

if os.path.isfile('saves/' + filename +'_lemmatized_features.pkl') and os.path.isfile('saves/' + test_filename +'_lemmatized_features.pkl'):
	print "Lemmatized words found, loading..."
	data = joblib.load('saves/' + filename +'_lemmatized_features.pkl')
	test_data = joblib.load('saves/' + test_filename +'_lemmatized_features.pkl')
	print "Done loading."
else:
	print "Lemmatized words not found, creating and saving..."
	with open(filename, 'rU') as f:
		reader = csv.reader(f)
		data = list(list(elem) for elem in csv.reader(f, delimiter=','))
	f.close()
	#remove headers from table
	data = data[1:] 

	with open(test_filename, 'rU') as f:
		reader = csv.reader(f)
		test_data = list(list(elem) for elem in csv.reader(f, delimiter=','))
	f.close()
	#remove headers from table
	test_data = test_data[1:] 

	def get_tokens(s):
	    retval = []
	    sents = sent_tokenize(s)
	    for sent in sents:
	        tokens = word_tokenize(sent)
	        retval.extend(tokens)
	    return retval

	#make all text lowercase and tokenize strings, then lemmatize words
	wnl = WordNetLemmatizer()
	for i in xrange(len(data)):
		data[i] = map(str,map(wnl.lemmatize, (word_tokenize(data[i][1].lower().translate(None, string.punctuation)))))
	for i in xrange(len(test_data)):
		test_data[i] = map(str,map(wnl.lemmatize, (word_tokenize(test_data[i][1].lower().translate(None, string.punctuation)))))

	joblib.dump(data, 'saves/' + filename +'_lemmatized_features.pkl')
	joblib.dump(test_data, 'saves/' + test_filename +'_lemmatized_features.pkl')
	print "Lemmatized words saved."

for i in xrange(len(data)):
	data[i] = Counter(data[i])
for i in xrange(len(test_data)):
	test_data[i] = Counter(test_data[i])

vec = FeatureHasher(n_features = 1000)
unigram = vec.fit_transform(data).toarray()
unigram_test = vec.transform(test_data).toarray()

print unigram.shape
print unigram_test.shape
joblib.dump(unigram, 'saves/' + filename +'_feature_vectors_' + 'UNIGRAMS' + '.pkl')
joblib.dump(unigram_test, 'saves/' + test_filename +'_feature_vectors_' + 'UNIGRAMS' + '.pkl')
print "Saved feature matrix to " + 'saves/' + filename +'_feature_vectors_' + 'UNIGRAMS' + '.pkl'
print "Saved test feature matrix to " + 'saves/' + test_filename +'_feature_vectors_' + 'UNIGRAMS' + '.pkl'








