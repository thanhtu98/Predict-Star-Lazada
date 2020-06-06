from utils import *
from scipy.sparse import coo_matrix
from sklearn.naive_bayes import MultinomialNB, BernoulliNB
from sklearn import tree
from sklearn.svm import SVC
from sklearn.ensemble import AdaBoostClassifier
from sklearn.metrics import accuracy_score , confusion_matrix


def evalutate(labels, predicts):
	acc = accuracy_score(labels, predicts) * 100
	print(confusion_matrix(labels, predicts))
	return acc


class MultiNB:
	def train(self, x_train, y_train, x_valid, y_valid):
		#print("Multinomial Naive Bayes")
		self.clf = MultinomialNB()
		self.clf.fit(x_train, y_train)
		#print("Train Set")
		preds = self.clf.predict(x_train)
		acc_train = evalutate(y_train, preds)
		#print("Test Set")
		preds = self.clf.predict(x_valid)
		acc_test = evalutate(y_valid, preds)
		return acc_train, acc_test
	
	def test(self, x_test):
		return self.clf.predict(x_test)

class BernouliNB:
	def train(self, x_train, y_train, x_valid, y_valid):
		#print("Bernoulli Naive Bayes")
		x_train[x_train > 0] = 1
		self.clf = BernoulliNB()
		self.clf.fit(x_train, y_train)
		#print("Train Set")
		preds = self.clf.predict(x_train)
		acc_train = evalutate(y_train, preds)
		#print("Test Set")
		preds = self.clf.predict(x_valid)
		acc_test = evalutate(y_valid, preds)
		return acc_train, acc_test
	
	def test(self, x_test):
		return self.clf.predict(x_test)
class DecisionTree:
	def train(self, x_train, y_train, x_valid, y_valid, max_depth = 30):
		#print("Classifier is decision tree")
		self.clf = tree.DecisionTreeClassifier(max_depth= max_depth) # best is 30
		self.clf.fit(x_train, y_train)
		#print("Train Set")
		preds = self.clf.predict(x_train)
		acc_train = evalutate(y_train, preds)
		#print("Test Set")
		preds = self.clf.predict(x_valid)
		acc_test = evalutate(y_valid, preds)
		return acc_train, acc_test, ""
	def test(self, x_test):
		return self.clf.predict(x_test)


class SVM:
	def train(self, x_train, y_train, x_valid, y_valid):
		#print("Classifier is Support Vector Machine kernel {}".format('linear'))
		self.clf = SVC(kernel= 'linear')
		
		self.clf.fit(x_train, y_train)
		#print("Train Set")
		preds = self.clf.predict(x_train)
		acc_train = evalutate(y_train, preds)
		#print("Test Set")
		preds = self.clf.predict(x_valid)
		acc_test = evalutate(y_valid, preds)
		return acc_train, acc_test
	def test(self, x_test):
		return self.clf.predict(x_test)

def ensemble(results, n_samples, n_classes = 5):
	votes = np.zeros((n_samples, n_classes + 1))
	for result in results:
		for i in range(n_samples):
			votes[i, result[i]] += 1
	pred = np.argmax(votes, axis=1)	
	return pred

class BoostingDecisionTree:
	def train(self, x_train, y_train, x_valid, y_valid):
		#print("Ada boosting with decision tree")
		self.clf = AdaBoostClassifier(tree.DecisionTreeClassifier(max_depth= 30), n_estimators= 50, 
					learning_rate= 0.55, algorithm='SAMME.R', random_state=1)
		
		self.clf.fit(x_train, y_train)
		print("Train Set")
		preds = self.clf.predict(x_train)
		acc_train = evalutate(y_train, preds)
		
		print("Test Set")
		preds = self.clf.predict(x_valid)
		acc_test = evalutate(y_valid, preds)
		
		return acc_train, acc_test, preds
	def test(self, x_test):
		return self.clf.predict(x_test)
class BoostingSVM:
	def train(self, x_train, y_train, x_valid, y_valid):
		#print("Ada boosting with SVM")
		self.clf = AdaBoostClassifier(SVC(kernel= 'linear'), n_estimators= 1, learning_rate= 0.5, algorithm='SAMME', random_state=1)

		
		self.clf.fit(x_train, y_train)
		#print("Train Set")
		preds = self.clf.predict(x_train)
		acc_train = evalutate(y_train, preds)
		
		#print("Test Set")
		preds = self.clf.predict(x_valid)
		acc_test = evalutate(y_valid, preds)
		
		return acc_train, acc_test
	def test(self, x_test):
		return self.clf.predict(x_test)



def k_fold(X, Y, clasifier, n_folds):
	X = X.toarray()
	nSamPerFold = int(np.floor(Y.shape[0] *1.0 / n_folds)) 

	folds = []
	acc_trains, acc_tests = [], []
	for i in range(n_folds):
		nIdx = min((i+1) * nSamPerFold , Y.shape[0])
		_X = X[i*nSamPerFold : nIdx, :]
		_Y = Y[i*nSamPerFold : nIdx]
		folds.append([_X, _Y])

	for i in range(n_folds):
		print("Start Fold {}".format(i))
		X_val, Y_val = folds[i][0], folds[i][1]
		idxs = list(range(n_folds))
		train_idx = idxs.remove(i)
		
		X_train, Y_train = [], []	
		for idx in idxs:
			x, y = folds[idx][0], folds[idx][1]
			X_train.append(x)
			Y_train.append(y)

		X_train, Y_train = np.concatenate(X_train, axis = 0), np.concatenate(Y_train, axis = 0)
		self.clf = clasifier()
		acc_train, acc_test = self.clf.train(X_train, Y_train, X_val, Y_val)
		print("Accuracy Train {}, Accuracy Test {}".format(acc_train, acc_test))
		acc_trains.append(acc_train)
		acc_tests.append(acc_test)

	print("Accuracy Train {}, Accuracy Test {}".format(np.mean(acc_trains), np.mean(acc_tests)))




	