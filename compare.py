from __future__ import division

file1 = open('testLSTM800.csv')
file2 = open('testLSTM400.csv')

l1 = file1.readlines()
l2 = file2.readlines()
# print l1[:10]
count = 0
for i in xrange(len(l1)):
	if l1[i] != l2[i]:
		count += 1

print "{} samples over {} (ratio {}%)".format(count, len(l1), count/len(l1) * 100)