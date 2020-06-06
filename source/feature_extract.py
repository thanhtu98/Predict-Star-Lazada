import numpy as np
from utils import *

def get_vocab(comments, n_grams):
	words = []
	
	for ci in range(len(comments)):
		comment = comments[ci]
		for wi in range(len(comment) - n_grams + 1):
			char = u"".join(comment[i] + u" " for i in range(wi, wi + n_grams))
			char = char.strip()
			words.append(char)
	vocab =  list(set(words))

	word2idx, idx2word = {}, {}
	for idx, char in enumerate(vocab):
		word2idx[char] = idx
		idx2word[idx] = char
	return vocab, word2idx, idx2word


def bag_of_word(comments, word2idx, n_grams):
	data = []
	for idx, comment in enumerate(comments):
		frequence = {}
		for wi in range(len(comment) - n_grams + 1):
			char = u"".join(comment[i] + u" " for i in range(wi, wi + n_grams))
			char = char.strip()
			if char in frequence.keys():
				frequence[char] += 1
			else:
				frequence[char] = 1

			for word in frequence.keys():
				if word in word2idx.keys():
					data.append([idx, word2idx[word], frequence[word]])
	return data



def unicode2number(labels):
	labels[0] = u'5'
	for idx, label in enumerate(labels):
		labels[idx] = int(label)
	return np.array(labels)


import math

def compute_IDF(comments, word2idx, n_grams):
	N = len(comments)
	idf_dict = {}
	idf_dict = dict.fromkeys(word2idx.keys(), 0)

	for comment in comments:
		words = []
		for idx in range(len(comment) - n_grams + 1):
			char = u"".join(comment[i] + u" " for i in range(idx, idx + n_grams)).strip()
			words.append(char)
		words = list(set(words))
		
		for word in words:
			idf_dict[word] += 1

	for word, count in idf_dict.iteritems():
		idf_dict[word] = math.log(N / float(count))

	return idf_dict


def compute_TF(comments, word2idx, n_grams):
	tf = []
	for comment in comments:
		words = []
		for idx in range(len(comment) - n_grams + 1):
			char = u"".join(comment[i] + u" " for i in range(idx, idx + n_grams)).strip()
			words.append(char)
		
		n_char = len(words)
		idf_dict = dict.fromkeys(list(set(words)), 0)
		
		for word in words:
			idf_dict[word] += 1.0 / n_char
		tf.append(idf_dict)
	return tf



def compute_TFIDF(comments, word2idx, n_grams):
	idf = compute_IDF(comments, word2idx, n_grams)
	tfs = compute_TF(comments, word2idx, n_grams)
	data = []
	for idx, tf in enumerate(tfs):
		for word in tf.keys():
			data.append([idx, word2idx[word], tf[word] * idf[word]])
	return data



ACRONYM_FILE = "../data/acronym.txt"
STOP_FILE = "../data/stop_word.txt"

import copy

def get_data(n_grams, TRAIN_FILE, TEST_FILE):
	train_data = read_file(TRAIN_FILE)[1:]
	test_data = read_file(TEST_FILE)

	acronym = read_file(ACRONYM_FILE)
	stop = read_file(STOP_FILE)

	train_labels, train_comments = seperate_label_comment(train_data)
	valid_labels, valid_comments = seperate_label_comment(test_data)



	acronym_dict = get_acronym_dict(acronym)
	stop_word = get_stop_word(stop)

	comments = train_comments + valid_comments

	_comments = pre_process(copy.deepcopy(comments), acronym_dict, stop_word)

	vocab, word2idx, idx2word = get_vocab(_comments, n_grams)
	data = compute_TFIDF(_comments, word2idx, n_grams)

	return np.array(data), unicode2number(train_labels), comments, word2idx













