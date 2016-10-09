def write_csv(file, pred):
	final_label = []
	for x in pred:
		if x == 0:
			final_label.append('math')
		elif x == 1:
			final_label.append('stat')
		elif x == 2:
			final_label.append('cs')
		elif x == 3:
			final_label.append('physics')
		else:
			raise Exception()

	# assert len(final_label) == pred.size

	outfile = open(file, 'w')
	for i in xrange(len(final_label)):
		final_label[i] = '{},{}\n'.format(i,final_label[i])
	final_label.insert(0,'id,category\n')
	outfile.writelines(final_label)