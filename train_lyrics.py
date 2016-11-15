''' Modules to read data from mysql, convert into a form which is pare-able by python, and then train data'''
import MySQLdb
from nltk.stem.porter import PorterStemmer
from sklearn.linear_model import LogisticRegression
from textblob import TextBlob

logistic = LogisticRegression()

# NOTE: Make sure mysql server has been started! 
# > mysql.server start
print 'will establish connection to db'
db = MySQLdb.connect(host="localhost", db="cs221_nlp", read_default_file='~/.my.cnf')
db_cursor = db.cursor()

# Format in which to put the songs:
# genre, url, lyrics
def get_songs_by_genre(genre_of_interest):
	query = "select lyrics, genre from song where genre = '%s'" %genre_of_interest
	db_cursor.execute(query)
	data = db_cursor.fetchall()
	percent80 = int(0.8*len(data))
	train = data[:percent80]
	test = data[percent80:]
	return (train, test)

# convert to a format that the algo can use
# https://www.codementor.io/python/tutorial/data-science-python-r-sentiment-classification-machine-learning
def feature_parse(data, option):
	# perform basic stuff, like stemming
	stemmer = PorterStemmer()
	data = stemmer.stem(data)
	if option == 'bow':
		# perform bag of words feature extraction
		return re.findall(r"\w+(?:[-']\w+)*|'|[-.(]+|\S\w*", data)		# http://www.nltk.org/book/ch03.html 'Regular expressions for tokenizing text'
	if option == '3gram':
		# 3-gram model (TODO: Including backoff?)
		raise Exception('3-gram has not been implemented yet!')


# Now train an algo as a binary classification thing on this set
