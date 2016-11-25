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

# sentence count
def sentence_stats(song_string):
	# input a string representation of a song's lyrics
	stats = defaultdict(int)
	stats['num_verses'] = song_string.count('\n\n')
	sentences = song_string.split('\n')
	stats['num_sentences'] = len(sentences) - stats['num_verses'] + 1
	stats['avg_words_per_sentence'] = sum(s.count(' ') + 1 for s in sentences)/float(len(sentences))
	stats['num_words'] = stats['avg_words_per_sentence']*stats['num_sentences']
	stats['avg_word_length'] = sum(len(s) for s in sentences)/stats['num_words']

	# return stats
	output = [stats[x] for x in stats.keys()]
	return output

# return the N most commonly used words in a group of songs entered
def NmostCom(dataset, N):
	# input is a dataframe with 'lyrics' and 'genre' as headers
	# Returns a defaultdict, with avg no of times word appears in each song in the dataset
	import operator
	from stop_words import get_stop_words
	stop_words = get_stop_words('en')

	L = len(dataset)
	words = defaultdict(float)
	for song in dataset['lyrics'].tolist():
		considered_words = set(song.split()) - set(stop_words)
		for w in considered_words:
			words[w] += 1./L
	# sorting will create a new list of tuples
	sorted_list = sorted(words.items(), key=operator.itemgetter(1))[-N:]
	return sorted_list

# type/token ratio for different genres

# bag of words conversion
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

# n-gram extraction
def ngram(song, n = 2):
	"""
	@param n: n in n-gram, default 2
	@param song: string input, song lyrics
	Returns a generator expression for tuples containing n-grams
	"""
	ngrams = nltk.ngrams(song.split(), n)
	return ngrams

def tupleify(dataset, twolists=False):
	# converts a dataframe (2xN) to a list of tuples

	l = dataset['lyrics'].tolist()
	g = dataset['genres'].tolist()
	output = (l,g)
	if twolists == True:
		# convert output to a list of tuples and not a tuple of lists
		output = []
		for i, lric in enumerate(l):
			output.append(lric, g[i])
	return output


# Caluclating performance for baseline
if __name__ == "__main__":
	pass




