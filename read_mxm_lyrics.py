# Read the data in mxm_lyrics_train.txt
# As given on http://labrosa.ee.columbia.edu/millionsong/musixmatch
import random
from collections import defaultdict
# from nltk.classify import NaiveBayesClassifier
from sklearn.ensemble import RandomForestClassifier
forest = RandomForestClassifier(n_estimators = 10) 	# TODO: What difference does changing n_estimators make
from sklearn.feature_extraction import DictVectorizer	# converts dict to a vector for sklearn predictors
import numpy

# hardcode the genre vector
genre_labels = ['Reggae', 'Latin', 'RnB', 'Jazz', 'Metal', 'Pop', 'Punk', 'Country', 'New Age', 'Rap', 'Rock', 'World', 'Blues', 'Electronic', 'Folk']

''' ---------------------------------------------- FUNCTION DEFINITIONS ---------------------------------------------- '''
def predict_genre(weights, x):
	"""
	@param list weights: A list of dict objects with the weights for each genre
	@param defaultdict x: Feature vector of the song (i.e. occurences of words)
	This function returns the predicted genre as a string, i.e. one of
	['Reggae', 'Latin', 'RnB', 'Jazz', 'Metal', 'Pop', 'Punk', 'Country', 'New Age', 'Rap', 'Rock', 'World', 'Blues', 'Electronic', 'Folk']
	Which is the one which gives the max dotProduct with the corresonding genre's weight
	"""
	_, i = max((dotProduct(weight, x), i) for i, weight in enumerate(weights))
	return genre_labels[i]

def read_genres(filename):
	"""
	@param filename: The location of the file containing genre information. Should have each line of the form:
	TRAAAAK128F9318786	Rock
	TRAAAFD128F92F423A      Punk    Rock
	Where the first is track_id and the second is the Genre
	Reads the file containing genre clasifications, returns a dict containing track_id <--> genre mappings
	One track_id can be mapped to multiple genres, these are entered as a list
	**NOTE: The header and endline is hardcoded
	"""
	print "reading genre classifications ...",
	# read in all genre classifications into a dict 
	# 15 genres are 'Reggae', 'Latin', 'RnB', 'Jazz', 'Metal', 'Pop', 'Punk', 'Country', 'New Age', 'Rap', 'Rock', 'World', 'Blues', 'Electronic', 'Folk'
	genre = {}
	f = open(filename, 'r')
	for _ in range(7,280838):	# skip header of 7 lines, go till end of 280838 lines
	  data = f.readline().strip()
	  data = data.split('\t')
	  genre[data[0]] = tuple(x for x in data[1:])		# put the data into the genre dict
	f.close()
	print "done!"
	return genre

def extract_data(words, start, end, f, song_list):
	"""
	Returns song data as given in the file format - track_id, 
	"""
	for i in range(start, end):
		d = defaultdict(int)
		ligne = f.readline()
		ligne = ligne.split(',')
		track_id = ligne[0]
		mxm_id = ligne[1]
		for j in range(2,len(ligne)):
		  # put rest of the line into the defaultdict
		  wordclass = ligne[j]
		  w_no, w_count = map(int, wordclass.split(':'))
		  d[words[w_no-1]] = w_count	# word index starts from 1!
		# append the (track_id, mxm_id, defaultdict) to the list of songs
		song_list.append((track_id, mxm_id, d))

def prepare_set(genre, songs):
	"""
	@param genre: A dict of genres, as returned by read_genres()
	@param songs: A list of songs, as returned by extract_data()
	Combines the genres and songs into a form which is parse-able by NLTK, which is a list of tuples (song_dict, genre)
	"""
	tset = []
	for song in songs:
		try:
			for g in genre[song[0]]:
				tset.append((song[2], g))	# take the first ([0]'th) genre of the genres that the current song identifies with
		except KeyError:
			continue
	return tset

def read_data_BoW():
	"""
	Read in files from the dataset and put them into a (training_set, testing_set) tuple
	Split is about 80/20
	Each of them is a dict of (bag_of_words, genre), where bag_of_words is a dict of words and no of occurences of the word
	genre is a string with the genre into which the song is classified
	"""
	f_train = open('mxm_dataset_train.txt', 'r')

	for _ in range(17):
	  # ignore first 17 header lines
	  f_train.readline()
	words = f_train.readline()
	words = words.split(',')

	# read songs into sparse vector
	print "reading songs for training...",
	training_songs = []	# list of songs, each represented by a defaultdict
	extract_data(words, 18, 210537, f_train, training_songs)
	f_train.close()
	print "done!",

	# Now combine both into a training set
	training_set = prepare_set(genre, training_songs)

	print "reading songs for testing...",
	f_test = open('mxm_dataset_test.txt', 'r')
	for _ in range(18):
	  # ignore first 18 header lines
	  f_test.readline()

	testing_songs = []	# list of songs, each represented by a defaultdict
	extract_data(words, 18, 27161, f_test, testing_songs)
	f_train.close()

	# Now combine both into a training set
	testing_set = prepare_set(genre, testing_songs)

	# return both train and test sets as tuple
	return (training_set, testing_set)

''' ------------------------------------------------------------------------------------------------ '''

# Now will use nltk to train a model
genre = read_genres('msd_tagtraum_cd2.cls')
train_set, test_set = read_data_BoW()
# TODO: Clean up dataset as given on https://www.kaggle.com/c/word2vec-nlp-tutorial/details/part-1-for-beginners-bag-of-words

# STRATEGY 1: Train using RandomForestClassifier (from sklearn)
# train on training set - similar to https://www.kaggle.com/c/word2vec-nlp-tutorial/details/part-1-for-beginners-bag-of-words
t = numpy.asarray(train_set)	# convert to a numpy array of dicts and strings
X = v.fit_transform(t[:,0])		# convert numpy array (1st col) to vector, has shape (142671, 5000)
forest = forest.fit(X, t[:,1])	# TRAIN
# then prepare the test set, first convert to numpy arry and then generate matrix from sparse vectors
test = numpy.asarray(test)	
Xt = v.fit_transform(test[:,0]) 
forest.predict(Xt)	# <---- this doesn't work out because the matrix Xt has a shape (19028, 4998) and the required no of features is 5000 

# STRATEGY 2: sNaive Bayes using NLTK
classifier = NaiveBayesClassifier.train(train_set)
print 'accuracy :', nltk.classify.util.accuracy(classifier, test_set)

# FURTHER READING: http://www.nltk.org/book/ch01.html
