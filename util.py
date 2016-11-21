""" ----------------------------------------------------------------
	Useful functions that will be referred to in the process of feature extraction and generation of statistics about a song lyric.

---------------------------------------------------------------- """
# extract statistics about a song
import pandas
from collections import defaultdict
import math


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

def logistic(weights, x):
	"""
	@param weights: The weights vector trained on a particular binary class
	@param x: the feature vector reprsentation of your input datapoint
	Returns the logistic function (logistic(z) = 1/(1 + exp(-z))) of dotProduct(weights, x)
	"""
	return 1./(1 + math.exp(-dotProduct(weights, x)))

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

# read in train and test data - to make it easy to debug on python terminal
def prepare_data(filename, num_datapoints=None, featureExtractor=bag_of_words):
	"""
	@param filename: The path to CSV file containing the data (pandas format)
	@param featureExtractor: (optional) The feature extractor you want to use. Default is bag of words
	@param num_datapoints: (optional) The number of samples to consider 
							(useful when debugging with small training sample)
	Returns a list of (feature, class) tuples.
	"""
	train_data = pandas.read_csv(filename)
	train_data = train_data.to_records(index=False)		# Now is a list of tuples (lyrics, genre)
	train_data = [(featureExtractor(l), g) for i,l,g in train_data]	
	if not num_datapoints == None:
		train_data = train_data[:num_datapoints]
	return train_data

# Caluclating performance for baseline
if __name__ == "__main__":
	# pandas also adds the index of the row, will be removed in this process
	train_data = read_data('train.csv')

	# train stochastic gradient descent on this, get weights
	genre_labels = ['Rock', 'Pop', 'Hip Hop/Rap', 'R&B;', 'Electronic', 'Country', 'Jazz', 'Blues', 'Christian', 'Folk']
	w = stochastic_grad_descent(train_data[:10], genre_labels)

	# Next, find precision recall for all these
	test_data = read_data('test.csv')
	sgd_performance(w, test_data[:10], genre_labels)




