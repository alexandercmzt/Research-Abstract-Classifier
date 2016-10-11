from __future__ import division

import argparse

parser = argparse.ArgumentParser()
parser.add_argument('files', nargs=2)
args = parser.parse_args()

file1 = open(args.files[0])
file2 = open(args.files[1])

l1 = file1.readlines()
l2 = file2.readlines()

count = 0
for i in xrange(len(l1)):
	if l1[i] != l2[i]:
		count += 1

print "{} samples over {} (ratio {}%)".format(count, len(l1), count/len(l1) * 100)