"""
Useful functions that will be referred to in the process of feature extraction and generation of statistics about a song lyric.
"""
# extract statistics about a song
import pandas
from collections import defaultdict
import nltk
import random
dataset = pandas.read_csv('lyrix.csv')	# data set


def increment(d1, scale, d2):
    """
    Implements d1 += scale * d2 for sparse vectors.
    @param dict d1: the feature vector which is mutated.
    @param float scale
    @param dict d2: a feature vector.
    [Acknowledgements: Extremely useful function taken from CS 221 hw2]
    """
    for f, v in d2.items():
        d1[f] = d1.get(f, 0) + v * scale


def dotProduct(d1, d2):
    """
    @param dict d1: a feature vector represented by a mapping from a feature (string) to a weight (float).
    @param dict d2: same as d1
    @return float: the dot product between d1 and d2
    Acknowledgements: Extremely useful function taken from CS 221 hw2]
    """
    if len(d1) < len(d2):
        return dotProduct(d2, d1)
    else:
        return sum(d1.get(f, 0) * v for f, v in d2.items())

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
def stochastic_grad_descent(training_set, genre_labels, numIters=10, eta=0.01):
	'''
	@param training_set: List of (song, genre)
	@param genres: list of genres used
	@param numIters: number of iterations to run, 10 is usually more than enough
	@param eta: step size for stochastic gradient descent, can play around to find optimal
	Given training_set, which is a list of (track_id, mxm_id, vector) tuples. 
	The 'vector' is a sparse vector containing number of times a word occurs
	in a song's lyrics.
	This function will return the weight vector (sparse
	feature vector) learned using stochastic gradient descent.
	'''
	weights = [{} for _ in range(len(genre_labels))]
	D = len(training_set)
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
		     lossFunc += loss(song[0], genre_labels[song[1]], weights)/D
		     # choose random vector element and update the gradient for that
		     random_song = random.sample(training_set, 1)[0]		# returns a list of single tuple, need to extract that tuple
		     increment_weight(random_song[0], genre_labels[random_song[1]], weights)
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


# Caluclating performance for baseline
if __name__ == "__main__":
	train_data = pandas.read_csv('train.csv')
	train_data = train_data.to_records(index=False)		# Now is a list of tuples (lyrics, genre)
	train_data = [(bag_of_words(l), g) for i,l,g in train_data]		# pandas also adds the index of the row, will be removed in this process


