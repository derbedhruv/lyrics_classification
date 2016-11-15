''' Modules to read data from mysql, convert into a form which is pare-able by python, and then train data'''
import MySQLdb
from nltk.stem.porter import PorterStemmer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.cross_validation import train_test_split
from textblob import TextBlob
import pandas as pd
import re

logistic = LogisticRegression()

# NOTE: Make sure mysql server has been started! 
# > mysql.server start
print 'will establish connection to db'
db = MySQLdb.connect(host="localhost", db="cs221_nlp", read_default_file='~/.my.cnf')
db_cursor = db.cursor()

# We fix upon 10 broad genres
genres = ['Rock', 'Pop', 'Hip Hop/Rap', 'R&B;', 'Electronic', 'Country', 'Jazz', 'Blues', 'Christian', 'Folk']

# Format in which to put the songs:
# genre, url, lyrics
def get_songs_by_genre(genre_of_interest):
	"""
	@param: genre_of_interest: The genre you are interested in, as a string
	Send query to db for a particular genre
	"""
	query = "select lyrics from song where genre = '%s'" %genre_of_interest
	db_cursor.execute(query)
	data = db_cursor.fetchall()
	return data

def get_data(genres=genres):
	"""
	Gets data from the db, arranges it in the form ('lyrics', genre_class), where genre_class is an int representing the genre
	It correspondds to the index in the genres list
	"""
	dataset = []
	for label, genre in enumerate(genres):
		data = get_songs_by_genre(genre)
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

vectorizer = CountVectorizer(
    analyzer = 'word',
    tokenizer = feature_parse,
    lowercase = True,
    stop_words = 'english'
)

# Now train an algo as a binary classification thing on a reduced subset
if __name__ == "__main__":
	dataset = get_data(genres=['Electronic', 'Country', 'Jazz', 'Blues', 'Christian', 'Folk'])

	data_features = vectorizer.fit_transform(dataset['lyrics'].tolist())
	data_features = data_features.toarray()
	# print vectorizer.get_feature_names()

	# Now we prepare everything for logistic regression
	X_train, X_test, y_train, y_test  = train_test_split(
        data_features, 
        dataset['genre'],
        train_size=0.85
    )
	LogisticRegressionClassifier = LogisticRegression()
	LogisticRegressionClassifier = LogisticRegressionClassifier.fit(X=X_train, y=y_train)

	# print how well classification was done
	y_pred = LogisticRegressionClassifier.predict(X_test)
	from sklearn.metrics import classification_report
	print(classification_report(y_test, y_pred))
