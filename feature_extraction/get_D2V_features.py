import csv 
import sys, os, codecs
from os import listdir
from os.path import isfile, join
import numpy as np
import sklearn
import string
from random import shuffle
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.externals import joblib
from gensim import utils
from gensim.models import Doc2Vec
from gensim.models.doc2vec import LabeledSentence

if len(sys.argv) != 3:
	print "Please use the path to the train.csv file as the first command line argument and the path to the test.csv file as the second command line argument."
filename = sys.argv[1]
test_filename = sys.argv[2]
#If feature vectors are already on disk, load them
if os.path.isfile('saves/data/' +'model_d2v'):
	print "Saved D2V model found, loading..."
	model = Doc2Vec.load('saves/data/' +'model_d2v')
	data = joblib.load('saves/' + filename +'_lemmatized_features.pkl')
	test_data = joblib.load('saves/' + test_filename +'_lemmatized_features.pkl')
	"Done loading"
#If not, we need to make them
else:
	#If lemmatized versions of csv text are already saved, load it
	print "Saved D2V model not found, creating and saving..."
	if os.path.isfile('saves/' + filename +'_lemmatized_features.pkl') and os.path.isfile('saves/' + test_filename +'_lemmatized_features.pkl'):
		print "Lemmatized words found, loading..."
		data = joblib.load('saves/' + filename +'_lemmatized_features.pkl')
		test_data = joblib.load('saves/' + test_filename +'_lemmatized_features.pkl')
		print "Done loading."
	#Otherwise lemmatize the csv file, tokenize, and lowercase
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
		joblib.dump(data, 'saves/' + test_filename +'_lemmatized_features.pkl')
		print "Lemmatized words saved."

	#Use lemmatized data to generate vectors
	class LabeledLineSentence(object):
	    def __init__(self, doc_list, labels_list=None):
	       self.doc_list = doc_list
	       #print "DOC LIST SIZE: " + str(len(doc_list))
	    def __iter__(self):
	        for idx, doc in enumerate(self.doc_list):
	            yield LabeledSentence(doc,[str(idx)])


	sentences = LabeledLineSentence(data + test_data)
	model = Doc2Vec(size=400, window=10, min_count=5, workers=11,alpha=0.025, min_alpha=0.025)
	print "Building vocabulary for D2V..."
	model.build_vocab(sentences)
	for epoch in range(10):
		print "Training round " + str(epoch+1) + "/10..."
		model.alpha -= 0.002
		model.min_alpha = model.alpha
		model.train(sentences)
	print "Done training. Saving D2V model..."
	model.save('saves/data/' +'model_d2v')
	print "D2V model saved."

output_matrix = []
for i in xrange(len(data)):
	output_matrix.append(model.docvecs[str(i)])
test_output_matrix = []
for j in xrange(i,i+len(test_data)):
	test_output_matrix.append(model.docvecs[str(j)])


print np.array(output_matrix).shape
print np.array(test_output_matrix).shape
joblib.dump(np.array(output_matrix), 'saves/' + filename +'_feature_vectors.pkl')
joblib.dump(np.array(test_output_matrix), 'saves/' + test_filename +'_feature_vectors.pkl')
print "Saved feature matrix to " + 'saves/' + filename +'_feature_vectors.pkl'
print "Saved test feature matrix to " + 'saves/' + test_filename +'_feature_vectors.pkl'




