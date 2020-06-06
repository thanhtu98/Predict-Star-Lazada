import numpy as np



def analysis(comments, preds, labels):
	flags = (preds == labels)
	print(np.where(flags == False))
	for idx, flag in enumerate(flags):
		if  not flag:
			print("Label {}, Predict {}".format(labels[idx], preds[idx]))
			print(comments[idx])


def label_distr(labels):
	ls = np.unique(labels)
	for l in ls:
		n  = np.sum(labels == l)
		print("{} distribution {}".format(l, n * 1.0 / labels.shape[0]))
		print("number sample {}".format(n))
