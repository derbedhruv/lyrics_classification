"""
Useful functions that will be referred to in the process of feature extraction and generation of statistics about a song lyric.
"""
# extract statistics about a song
import pandas
from collections import defaultdict
import nltk
dataset = pandas.read_csv('lyrix.csv')	# data set

# 1. sentence count
def sentence_count(dataset):
	genre = {}
	for i, song in enumerate(dataset['lyrics']):
	  g = song[1]
	  num_verses = lyric.count('\n\n')
	  num_sentences = lyric.count('\n') - num_verses + 1

# 2. type/token ratio for different genres

# 3. bag of words conversion
def bag_of_words(song):
	"""
	@param song: A string, corresponding to all or part of a song's lyrics
	Returns a defaultdict with the bag of words representation of the song
	"""
	bow = defaultdict(int)
	song = song.split()
	for word in song:
		bow[word] += 1
	return bow

# 4. n-gram extraction
def ngram(song, n = 2):
	"""
	@param n: n in n-gram, default 2
	@param song: string input, song lyrics
	Returns a generator expression for tuples containing n-grams
	"""
	ngrams = nltk.ngrams(song.split(), n)
	return ngrams

# 5. The old stochastic gradient descent for all-vs-one classification
def stochastic_grad_descent(training_set, genres, numIters=10, eta=0.01):
	'''
	@param training_set: 
	@param genres:
	@param numIters: number of iterations to run, 10 is usually more than enough
	@param eta: step size for stochastic gradient descent, can play around to find optimal
	Given training_set, which is a list of (track_id, mxm_id, vector) tuples. 
	The 'vector' is a sparse vector containing number of times a word occurs
	in a song's lyrics.
	This function will return the weight vector (sparse
	feature vector) learned using stochastic gradient descent.
	'''
	weights = [{} for _ in range(len(genre_labels))]
	D = len(training_songs)
	random.seed(88)
	def loss(xx, yy, weights):
		# the hinge loss in 0-1 prediction of xx as yy
		out = 0
		for i, weight in enumerate(weights):
		     # return 1 if it is the genre corresponding to this weight, else return -1
		     if yy[0] == genre_labels[i]:
		             y = 1
		     else:
		             y = -1
		     # find hinge loss for each genre vector
		     out += max(0, 1 - y*dotProduct(xx, weight))
		return out
         		
	def increment_weight(xx, yy, weights):
		# use the increment() function to make things convenient
		for i, weight in enumerate(weights):
			# return 1 if it is the genre corresponding to this weight, else return -1
			if yy[0] == genre_labels[i]:
				y = 1
			else:
				y = -1
			if y*dotProduct(weight, xx) < 1:
				increment(weight, eta*y, xx)
	for i in range(numIters):
		# calculate loss function with current vector 'weights'
		lossFunc = 0
		for song in training_set:
		 try:
		     lossFunc += loss(song[2], genres[song[0]], weights)/D
		     # choose random vector element and update the gradient for that
		     random_song = random.sample(training_set, 1)[0]
		     increment_weight(random_song[2], genres[random_song[0]], weights)
		 except KeyError:
		         # skip that example
		         pass
		# print "i = ",i,", loss = ", lossFunc
	return weights

def predict_genre_sgd(weights, x, genre_labels):
	"""
	@param list weights: A list of dict objects with the weights for each genre
	@param defaultdict x: Feature vector of the song (i.e. occurences of words)
	This function returns the predicted genre as a string, i.e. one of genre_labels
	Which is the one which gives the max dotProduct with the corresonding genre's weight
	"""
	_, i = max((dotProduct(weight, x), i) for i, weight in enumerate(weights))
	return genre_labels[i]

def sgd_performance(weights, testdata):
	"""
	@param list weights: A list of dict objects with the weights for each genre
	@param testdata: The testdata in the form
	"""
