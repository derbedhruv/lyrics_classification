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
from nltk.stem.porter import PorterStemmer
from textblob import TextBlob
import re
import classifiers
import util

valid_cli_args = ['-d', '-f', '--baseline', '--rfc']
# We fix upon 10 broad genres
genre_labels = ['Rock', 'Pop', 'Hip Hop/Rap', 'R&B;', 'Electronic', 'Country', 'Jazz', 'Blues', 'Christian', 'Folk']

# Format in which to put the songs:
# genre, url, lyrics
def get_songs_by_genre(genre_of_interest, db_cursor):
	"""
	@param: genre_of_interest: The genre you are interested in, as a string
	Send query to db for a particular genre
	"""
	import MySQLdb
	query = "select lyrics from song where genre = '%s'" %genre_of_interest
	db_cursor.execute(query)
	data = db_cursor.fetchall()
	return data

def get_data(genres=genre_labels):
	"""
	@param genres: List of genres (strings) to collect from the db
	Gets data from the db, arranges it in the form ('lyrics', genre_class), where genre_class is an int representing the genre
	It correspondds to the index in the genres list. Returns a Panda object (dataframe).
	"""
	print 'establishing connection to db...',
	db = MySQLdb.connect(host="localhost", db="cs221_nlp", read_default_file='~/.my.cnf')
	db_cursor = db.cursor()
	print 'done!'
	dataset = []
	for label, genre in enumerate(genres):
		data = get_songs_by_genre(genre, db_cursor)
		for song in data:
			# song is a singleton tuple (since the db query returns only tuples), so need to extract
			dataset.append([song[0], label])
	# convert to pandas
	dataset = pd.DataFrame.from_records(dataset, columns=['lyrics', 'genre'])
	return dataset

# convert to a format that the algo can use
# REF: https://www.codementor.io/python/tutorial/data-science-python-r-sentiment-classification-machine-learning
def feature_parse(data, option='bow'):
	"""
	The data is a tuple of tuples (lyrics, genre)
	We want to clean this up and make it into a numpy array
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

def train_logistic(dataset):
	"""
	@param dataset: DataFrame containing ('lyrics', genre) where genre is an integer class 0..N 
	trains a logistic regression classifier and reports how well it performs on a cross-validation dataset.
	returns the fitted classifier object (sklearn.linear_model.LogisticRegression).
	REF: https://www.codementor.io/python/tutorial/data-science-python-r-sentiment-classification-machine-learning
	"""
	# create an instance of CountVectorizer class which will vectorize the data
	vectorizer = CountVectorizer(
	    analyzer = 'word',
	    tokenizer = feature_parse,
	    lowercase = True,
	    stop_words = 'english',
	    max_features = 3000		# can go upto 5000 on corn.stanford.edu
	)
	# Fit the data
	data_features = vectorizer.fit_transform(dataset['lyrics'].tolist())
	data_features = data_features.toarray()
	# print vectorizer.get_feature_names()

	# Now we prepare everything for logistic regression
	X_train, X_test, y_train, y_test  = train_test_split(
        data_features, 
        dataset['genre'],
        train_size=0.80
    )
	LogisticRegressionClassifier = LogisticRegression()
	LogisticRegressionClassifier = LogisticRegressionClassifier.fit(X=X_train, y=y_train)

	# print how well classification was done
	y_pred = LogisticRegressionClassifier.predict(X_test)
	print(classification_report(y_test, y_pred))

	return LogisticRegressionClassifier


def get_traintest(n=None):
	"""
	Returns standard train and test set
	"""
	train_data = util.prepare_data('train.csv', num_datapoints=n)
	test_data = util.prepare_data('test.csv', num_datapoints=n)
	return (train_data, test_data)

def get_features(dataset):
	# TODO: Add features/vectorizer to this, modularize code!
	return None

if __name__ == "__main__":
	# Check command line args...
	option = str(sys.argv[1])
	assert option in valid_cli_args, '%s is not a valid argument!'%option
	if option == '-d':
		print 'Will grab data from MYSQL database..'
		# NOTE: Make sure mysql server has been started! 
		# > mysql.server start
		dataset = get_data(genres=['Electronic', 'Country', 'Jazz', 'Blues', 'Christian', 'Folk'])
		train_logistic(dataset)
	if option == '-f':
		try:
			filename = str(sys.argv[2])
			print 'Will grab data from %s..'%filename,
		except IndexError:
			print 'ERROR: Please enter a file location to get the data from! Command is -f </path/to/file>. Quitting...'
			sys.exit(0)
		# Read data from the CSV file and call the classifier to train on it
		dataset = pd.read_csv(filename)
		print 'read complete. Training...'
		# train_logistic(dataset)
		# train_naiveBayes(dataset)
		# trainRandomForest(dataset)		# READ https://www.stat.berkeley.edu/~breiman/RandomForests/cc_home.htm
		# trainNeuralNet(dataset)
	if option == '--baseline':
		print 'training the baseline..'
		train_data, test_data = get_traintest()

		# train stochastic gradient descent on this, get weights
		genre_labels = ['Rock', 'Pop', 'Hip Hop/Rap', 'R&B;', 'Electronic', 'Country', 'Jazz', 'Blues', 'Christian', 'Folk']
		baseline = classifiers.Baseline(train_data, genre_labels, debug=True)
		baseline.stochastic_grad_descent()
		baseline.saveModel('baseline-21Nov16.txt')

		# Next, find precision recall for all these
		util.performance(baseline.predict, test_data, genre_labels)

	if option == '--rfc':
		train_data, test_data = get_traintest(n=100)
		print 'training random forest..'
		genre_labels = ['Rock', 'Pop', 'Hip Hop/Rap', 'R&B;', 'Electronic', 'Country', 'Jazz', 'Blues', 'Christian', 'Folk']
		rfc = classifiers.RandomForestClassifier(train_data, num_trees=1, num_features=10, class_labels=genre_labels, tree_depth=4)
		rfc.create_random_forest()

		# evaluate performance
		util.performance(rfc.predict, train_data, genre_labels)


		
