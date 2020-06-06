from model import *
from utils import *
from feature_extract import *
from analysis import *
from sklearn.model_selection import train_test_split
import re
import argparse

parser = argparse.ArgumentParser(description='')
parser.add_argument('--train_file', dest='train_file',  default = "../data/train.txt")
parser.add_argument('--test_file', dest='test_file',  default = "../data/test.txt")
parser.add_argument('--save_file', dest='save_file', default = "Nhom_5_Submit_3.txt")
parser.add_argument('--model', dest='model',  default = "bdt")
args = parser.parse_args()


def write_submition(file_name, test_preds):
	test_content = read_file("../data/test.txt")
	f= open(file_name ,"w+")
	for i, content in enumerate(test_content):
		content = re.sub("-1", str(test_preds[i]), content) + "\n"
		f.write(content)
	f.close()



if __name__ == "__main__":
	X, Y, comments, word2idx = get_data(1, args.train_file , args.test_file)
	X = coo_matrix((X[:, 2], (X[:, 0], X[:, 1])), shape = (len(comments), len(word2idx.keys()))).toarray()
	x_train = X[0:Y.shape[0], :]
	x_valid = X[Y.shape[0] : , :]

	if args.model == "dt":
		model = DecisionTree()
		_, _, preds = model.train(x_train, Y, x_train, Y)
	elif args.model == "bdt":
		model = BoostingDecisionTree()
		_, _, preds = model.train(x_train, Y, x_train, Y)
	elif args.model == "svm":
		model = SVM()
		_, _  = model.train(x_train, Y, x_train, Y)

	test_preds = model.test(x_valid)
	write_submition(args.save_file, test_preds)

	test_content = read_file(args.save_file)
	print(len(test_content))