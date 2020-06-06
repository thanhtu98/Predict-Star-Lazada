# coding: utf-8

import os
import re
import numpy as np
import copy
from string import punctuation

def read_file(file_name):
	with open(file_name, "r") as file_content:
		data = file_content.read()
	data = data.strip()
	return data.split("\n")


def get_label_comment(line):
	line = unicode(line, "utf-8")
	words = line.split(u" ")
	label = words[0]
	for word in words[2:]:
		words[1] += (u" " + word)
	return label, words[1]


def seperate_label_comment(content):
	labels, comments = [], []
	for i in range(0, len(content)):
		label, comment = get_label_comment(content[i])
		labels.append(label)
		comments.append(comment)
	return labels, comments

def drop_pun(sen):
	punc_chars = list(set(punctuation))

	sen = re.sub(u"([!@#$%^&*<>?/:;])", u" ", sen)
	sen = re.sub(u" +", u" ", sen)
	sen = sen.lower()
	sen = sen.strip()
	return sen



def get_acronym_dict(acronyms):
	acronym_dict = {}
	for line in acronyms:
		line = unicode(line, "utf-8")
		[acronym, full] = line.split(u':')
		acronym_dict[acronym] = full
	return acronym_dict

def get_stop_word(stop_context):
	stop_word = []
	for line in stop_context:
		stop_word.append(unicode(line, "utf-8"))
	return stop_word


def pre_process(comments, acronyms, stop_word):
	lines = []
	for comment in comments:
		comment = drop_pun(comment)
		words = comment.split(u" ")
		approved_words = []
		for word in words:
			if word in acronyms.keys():
				word = acronyms[word]
			elif word in stop_word:
				continue
			else:
				approved_words.append(word)
		lines.append(approved_words)
	return lines
























# def get_vocab(language):
# 	words = copy.copy(language[0])
# 	for i in range(1, len(language)):
# 		words += language[i]
# 	vocab = list(set(words))
# 	word2idx, idx2word = {}, {}
# 	for idx, word in enumerate(vocab):
# 		word2idx[word] = idx
# 		idx2word[idx] = word
# 	return word2idx, idx2word

# def bag_of_word(comments, word2idx):
# 	data = []
# 	for idx, comment in enumerate(comments):
# 		frequence = {}
# 		for word in comment:
# 			if word in frequence.keys():
# 				frequence[word] += 1
# 			else:
# 				frequence[word] = 1
# 		for word in frequence.keys():
# 			if  word in word2idx.keys():
# 				data.append([idx, word2idx[word], frequence[word]])
# 	return data

# def unicode2number(labels):
# 	labels[0] = u'5'
# 	for idx, label in enumerate(labels):
# 		labels[idx] = int(label)
# 	return np.array(labels)


# import math

# def compute_IDF(comments, word2idx):
# 	N = len(comments)
# 	idf_dict = {}

# 	idf_dict = dict.fromkeys(word2idx.keys(), 0)
	
# 	for comment in comments:
# 		words = list(set(comment))
# 		for word in words:
# 			idf_dict[word] += 1

# 	for word, count in idf_dict.iteritems():
# 		idf_dict[word] = math.log(N / float(count))

# 	return idf_dict


# def compute_TF(comments, word2idx):
# 	tf = []
# 	for comment in comments:
# 		n_chac = len(comment)
# 		words = list(set(comment))
# 		idf_dict = dict.fromkeys(words, 0)
# 		for word in comment:
# 			idf_dict[word] += 1*1.0 / n_chac
# 		tf.append(idf_dict)
# 	return tf



# def compute_TFIDF(comments, word2idx):
# 	idf = compute_IDF(comments, word2idx)
# 	tfs = compute_TF(comments, word2idx)
# 	data = []
# 	for idx, tf in enumerate(tfs):
# 		for word in tf.keys():
# 			data.append([idx, word2idx[word], tf[word] * idf[word]])
# 	return data



# def preprocess2():
# 	train_file = "../data/train.txt"	
# 	valid_file = "../data/valid.txt"

# 	acronym =  "../data/acronym.txt"
# 	stop = "../data/stop_word.txt"

# 	train_data = read_file(train_file)[1:]
# 	valid_data = read_file(valid_file)[1:]

# 	train_data += valid_data


# 	acronym_characters = read_file(acronym)
# 	stop_word = read_file(stop)
	
# 	labels, comments = seperate_label_comment(train_data)
# 	_comments = comments

# 	acronym_dict = get_acronym_dict(acronym_characters)
# 	stop_word = get_stop_word(stop_word)
	
# 	comments = pre_process(comments, acronym_dict, stop_word)

# 	word2idx, idx2word = get_vocab(comments)

# 	data = compute_TFIDF(comments, word2idx)

# 	return np.array(data), unicode2number(labels), _comments, word2idx, idx2word



	






# def get_data():
# 	train_file = "../data/train.txt"
# 	valid_file = "../data/valid.txt"
	
# 	acronym =  "../data/acronym.txt"
# 	stop = "../data/stop_word.txt"
# 	train_data = read_file(train_file)[1:]
# 	# valid_data = read_file(valid_file)[1:]
# 	# train_data += valid_data
# 	data = train_data
# 	labels, comments = seperate_label_comment(data)
# 	_comments = copy.copy(comments)
# 	acronym_characters = read_file(acronym)
# 	stop_word = read_file(stop)
# 	acronym_dict = get_acronym_dict(acronym_characters)
# 	stop_word = get_stop_word(stop_word)
# 	comments = pre_process(comments, acronym_dict, stop_word)
# 	# word2idx, idx2word = get_vocab(comments)
# 	# train_data = bag_of_word(comments[0:], word2idx)
# 	# labels = unicode2number(labels)

# 	# return np.array(train_data), np.array(labels), _comments, word2idx, idx2word	


