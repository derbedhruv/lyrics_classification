"""
	Lyrics-based Music Classification
------------------------------------------------------------------------------------------------
	CS 221 Fall 2016 Project

	Author: Dhruv Joshi

	A modular system to train different algorithms to identify strings of lyrics and classify them into genres.
	10 most common genres have been selected, based on topicality, propensity to being identified through lyrics and availability

	Algorithms used are: 
	- Logistic Regression
	- Naive Bayes
	- Random Forest
	- Neural Networks

"""
''' Modules to read data from mysql, convert into a form which is pare-able by python, and then train data'''
import sys
import MySQLdb
from nltk.stem.porter import PorterStemmer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from textblob import TextBlob
import pandas as pd
import re

logistic = LogisticRegression()

# command line parameter variables
valid_cli_args = ['-f', '-m']
valid_models = ['log']

# We fix upon 10 broad genres
genres = ['Rock', 'Pop', 'Hip Hop/Rap', 'R&B;', 'Electronic', 'Country', 'Jazz', 'Blues', 'Christian', 'Folk']

# default file from which to read data
filename = 'songData-Nov22.csv'


# convert to a format that the algo can use
# REF: https://www.codementor.io/python/tutorial/data-science-python-r-sentiment-classification-machine-learning
def feature_parse(data, option='bow'):
	"""
	The data is a tuple of tuples (lyrics, genre)
	"""
	# perform basic stuff, like stemming
	# stemmer = PorterStemmer()
	# data = stemmer.stem(data)
	if option == 'bow':
		# perform bag of words feature extraction
		tokens = re.findall(r"\w+(?:[-']\w+)*|'|[-.(]+|\S\w*", data)		# REF: http://www.nltk.org/book/ch03.html 'Regular expressions for tokenizing text'
	if option == '3gram':
		# 3-gram model (TODO: Including backoff?)
		raise Exception('3-gram has not been implemented yet!')
	return tokens

def train_logistic(X_train, y_train):
	"""
	@param dataset: DataFrame containing ('lyrics', genre) where genre is an integer class 0..N 
	trains a logistic regression classifier and reports how well it performs on a cross-validation dataset.
	returns the fitted classifier object (sklearn.linear_model.LogisticRegression).
	REF: https://www.codementor.io/python/tutorial/data-science-python-r-sentiment-classification-machine-learning
	"""
	
	LogisticRegressionClassifier = LogisticRegression()
	LogisticRegressionClassifier = LogisticRegressionClassifier.fit(X=X_train, y=y_train)

	# print how well classification was done
	y_pred = LogisticRegressionClassifier.predict(X_test)
	print(classification_report(y_test, y_pred))

	return LogisticRegressionClassifier


def get_features(dataset, max_features=3000, tokenizer=feature_parse):
	"""
	Choose features and the way that features are created
	Create train test split from dataset after sparse feature representations are created
	"""
	# create an instance of CountVectorizer class which will vectorize the data
	vectorizer = CountVectorizer(
	    analyzer = 'word',
	    tokenizer = tokenizer,
	    lowercase = True,
	    stop_words = 'english',
	    max_features = max_features		# can go upto 5000 on corn.stanford.edu
	)
	# Fit the data
	data_features = vectorizer.fit_transform(dataset['lyrics'].tolist())
	data_features = data_features.toarray()
	# print vectorizer.get_feature_names()

	# tf-idf transformation
	# TODO: Put option for having or not having this
	tfidf_transformer = TfidfTransformer()
	X_train_tfidf = tfidf_transformer.fit_transform(data_features)

	# Now we prepare everything for logistic regression
	X_train, X_test, y_train, y_test  = train_test_split(
        data_features, 
        dataset['genre'],
        train_size=0.80
    )

    # Can extract a lot of features from the vectorizer
    # vectorizer.vocabulary_ gives the words used in the selected sparse feature matrix (http://stackoverflow.com/questions/22920801/can-i-use-countvectorizer-in-scikit-learn-to-count-frequency-of-documents-that-w)
    # 

	return (X_train, y_train, X_test, y_test, vectorizer)


def train_naiveBayes(dataset):
	"""
	@param dataset: DataFrame containing ('lyrics', genre) where genre is an integer class 0..N 
	Trains a Naive Bayes classifier and reports how well it performs
	REF: http://scikit-learn.org/stable/tutorial/text_analytics/working_with_text_data.html
	"""
	# create an instance of CountVectorizer class which will vectorize the data
	vectorizer = CountVectorizer(
	    analyzer = 'word',
	    tokenizer = feature_parse,
	    lowercase = True,
	    stop_words = 'english',
	    max_features = 3000
	)
	# fit bag of words model and convert to relative frequencies instead of absolute counts
	data_features = vectorizer.fit_transform(dataset['lyrics'].tolist())
	data_features = data_features.toarray()
	tfidf_transformer = TfidfTransformer()
	X_train_tfidf = tfidf_transformer.fit_transform(data_features)

	# Now we prepare everything for logistic regression
	X_train, X_test, y_train, y_test  = train_test_split(
        X_train_tfidf, 
        dataset['genre'],
        train_size=0.80
    )

	# train classifier
	naiveBayesClassifier = MultinomialNB().fit(X=X_train, y=y_train)
	# print how well classification was done
	y_pred = naiveBayesClassifier.predict(X_test)
	print(classification_report(y_test, y_pred))

	return naiveBayesClassifier 

def trainRandomForest(dataset):
	"""
	@param dataset: DataFrame containing ('lyrics', genre) where genre is an integer class 0..N 
	Trains a random forest classifier 
	https://www.stat.berkeley.edu/~breiman/RandomForests/cc_home.htm
	"""
	vectorizer = CountVectorizer(
	    analyzer = 'word',
	    tokenizer = feature_parse,
	    lowercase = True,
	    stop_words = 'english',
	    max_features = 3000
	)
	data_features = vectorizer.fit_transform(dataset['lyrics'].tolist())
	data_features = data_features.toarray()
	tfidf_transformer = TfidfTransformer()
	X_train_tfidf = tfidf_transformer.fit_transform(data_features)

	# Now we prepare everything for logistic regression
	X_train, X_test, y_train, y_test  = train_test_split(
        X_train_tfidf, 
        dataset['genre'],
        train_size=0.80
    )
	# random forest classifier with 100 trees
	forest = RandomForestClassifier(n_estimators = 100) 
	forest = forest.fit(X=X_train, y=y_train)

	y_pred = forest.predict(X_test)
	print(classification_report(y_test, y_pred))

	return forest 

def trainNeuralNet(dataset):
	"""
	@param dataset: DataFrame containing ('lyrics', genre) where genre is an integer class 0..N 
	Trains a neural network
	REF: http://scikit-learn.org/stable/modules/neural_networks_supervised.html
	"""
	vectorizer = CountVectorizer(
	    analyzer = 'word',
	    tokenizer = feature_parse,
	    lowercase = True,
	    stop_words = 'english',
	    max_features = 3000
	)
	data_features = vectorizer.fit_transform(dataset['lyrics'].tolist())
	data_features = data_features.toarray()
	tfidf_transformer = TfidfTransformer()
	X_train_tfidf = tfidf_transformer.fit_transform(data_features)

	# Now we prepare everything for logistic regression
	X_train, X_test, y_train, y_test  = train_test_split(
        X_train_tfidf, 
        dataset['genre'],
        train_size=0.80
    )

	# train NN, limited memory BFGS, step size 1e-5
	nn = MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(5, 2), random_state=1)
	nn = nn.fit(X=X_train, y=y_train)

	y_pred = nn.predict(X_test)
	print(classification_report(y_test, y_pred))

	return nn


def run_model(cl):
	"""
	@param cl: The command line index from which to consider the model command
	Second one has to be one of 
	"""
	try:
		cl1 = str(sys.argv[cl])
		cl2 = str(sys.argv[cl+1])
	except IndexError:
		print 'Please choose a model!'
		command_line_syntax()
		sys.exit(0)

	assert cl1 == '-m', 'You must enter -m to choose the model!'
	assert cl2 in valid_models, command_line_syntax()

	# First read in the data
	with open(filename, 'r') as f:
		dataset = pd.read_csv(f)

	# Then create the features
	X_train, y_train, X_test, y_test, vectorizer = get_features(dataset)

	# Then run models based on what the argument says
	if cl2 == 'log':
		logC = train_logistic(X_train, y_train)


def command_line_syntax(custom_starting_message=None):
	"""
	Tell user the correct syntax to use, then exit.
	"""
	if custom_starting_message:
		print custom_starting_message
	print 'Syntax of the command is:\npython main.py -f (optional)<file-to-get-data> -m <model-name>'
	print 'Options are\n\trfc - Random Forest Classifier\n\tbaseline - the baseline implementation\n\tnn - Neural Networks\n\tlog - Logstic Regression'
	print 'Quitting...'
	sys.exit(0)	


if __name__ == "__main__":
	# Check command line args...
	try:
		option = str(sys.argv[1])
		assert option in valid_cli_args, command_line_syntax('%s is not a valid argument!'%option)
	except IndexError:
		command_line_syntax('You have not entered any arguments!')

	if option == '-f':
		try:
			filename = str(sys.argv[2])
			print 'Will grab data from %s..'%filename,
		except IndexError:
			command_line_syntax('ERROR: Please enter a file location to get the data from!')
			sys.exit(0)
		# Read data from the CSV file and call the classifier to train on it
		# dataset = pd.read_csv(filename)
		print 'read complete.'

		# Now look for next argument for the type of classifier to use
		try:
			run_model(3)
		except IndexError:
			command_line_syntax('Please choose a model!')
			sys.exit(0)
	elif option == '-m':
		# first read default data
		run_model(1)

	command_line_syntax('No valid command line options given!')
	# train_logistic(dataset)
	# train_naiveBayes(dataset)
	# trainRandomForest(dataset)		# READ https://www.stat.berkeley.edu/~breiman/RandomForests/cc_home.htm
	# trainNeuralNet(dataset)



