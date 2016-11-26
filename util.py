""" ----------------------------------------------------------------
	Useful functions that will be referred to in the process of feature extraction and generation of statistics about a song lyric.

---------------------------------------------------------------- """
# extract statistics about a song
import pandas
from collections import defaultdict
import math
import pickle
import operator
import nltk
from stop_words import get_stop_words	# https://pypi.python.org/pypi/stop-words
stop_words = get_stop_words('en')

from main import genres
num_genres = len(genres)

# default values for important stuff
### ------------------------------------------------------------------------------------------####
def genres():
	return ['Rock', 'Pop', 'Hip Hop/Rap', 'R&B;', 'Electronic', 'Country', 'Jazz', 'Blues', 'Christian', 'Folk']
def filename():
	return 'songData-Nov25.csv'	# <------- ONLY CHANGE THIS, THE REST ARE DERIVED FROM IT
### ------------------------------------------------------------------------------------------####

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
	# stats = defaultdict(int)
	# stats['num_verses'] = song_string.count('\n\n')
	sentences = song_string.split('\n')
	words = song_string.split()
	# stats['num_sentences'] = len(sentences) - stats['num_verses'] + 1
	# stats['avg_words_per_sentence'] = sum(s.count(' ') + 1 for s in sentences)/float(len(sentences))
	# stats['num_words'] = len(words)
	# stats['avg_word_length'] = sum(len(w) for w in words)/len(words)

	stats = []
	# important numerical features
	stats.append(song_string.count('\n\n'))
	stats.append(len(sentences) - stats[0] + 1)		# stats[0] <--> no of verses
	stats.append(sum(s.count(' ') + 1 for s in sentences)/float(len(sentences)))
	stats.append(len(words))
	stats.append(sum(len(w) for w in words)/len(words))

	# next, get the number of occurences of top words
	topwords_dump_filename = 'TopWords_' + filename()
	with open(topwords_dump_filename) as f:
		topwords = pickle.load(f)

	# reduce only to useful words..
	topwords_only = [w for w in words if w in topwords]
	for tw in topwords:
		# stats['count_' + tw] += topwords_only.count(tw)
		stats.append(topwords_only.count(tw))

	# return stats
	# output = [stats[x] for x in stats.keys()]
	# return output
	return stats

''' the following functions have been used only once - for the preliminary analysis of songs '''
''' -----------------------------------------------------------------------------------------'''

# return the N most commonly used words in a group of songs entered
def NmostCom(dataset, N):
	# input is a dataframe with 'lyrics' and 'genre' as headers
	# Returns a defaultdict, with avg no of times word appears in each song in the dataset
	lyrics_set = dataset['lyrics'].tolist()
	genres_set = dataset['genre'].tolist()

	L = len(dataset)
	words = [defaultdict(float) for _ in range(num_genres)]

	for i, song in enumerate(lyrics_set):
		considered_words = set([s.lower() for s in song.split()]) - set(stop_words)
		for w in considered_words:
			words[genres_set[i]][w] += 1./L
	# sorting will create a new list of tuples
	sorted_list = range(num_genres)		# empty list with num_genres elements needed
	for i in range(num_genres):
		sorted_list[i] = sorted(words[i].items(), key=operator.itemgetter(1))[-N:]	# http://stackoverflow.com/questions/613183/sort-a-python-dictionary-by-value
	return sorted_list

# Find the 200 most common words in each genre,
# Then remove all common words for each genre and keep the 100 remaining for each
def save100MostComPerGenre():
	# Read in from same consistent filename used in main
	# Then call the NmostCom function
	dataset = pandas.read_csv(filename)
	mostCommon200Words = NmostCom(dataset, 200)

	# Then remove common words from all
	# convert to lists
	converted_list = range(num_genres)	# empty list with 10 elements initialized
	for j in range(num_genres):
		words_in_others = set([x[0] for i, y in enumerate(mostCommon200Words)for x in y if not i == j])
		exclusive_words_in_this_genre = set([x[0] for x in mostCommon200Words[j]])
		converted_list[j] = list(exclusive_words_in_this_genre - words_in_others)
		converted_list[j] = converted_list[j][:100]
	return converted_list

# Find the most common 400 words of all genres
def save400MostCom():
	dataset = pandas.read_csv(filename())
	mostCommon200Words = NmostCom(dataset, 200)

	# Now just make a set of all the distinct words
	setofwords = set(x[0] for y in mostCommon200Words for x in y)	# this is found to be 428 words on the Nov-22 dataset

	# dump the pickle
	dumpfile = 'TopWords_'+filename
	print 'saving to ', dumpfile
	with open(dumpfile, 'w') as f:
		pickle.dump(setofwords, f)

def NMostComNgrams(dataset, n):
	# n - the n-gram to consider
	# dataset - dataframe of ['lyrics', 'genre']
	lyrics_set = dataset['lyrics'].tolist()
	genres_set = dataset['genre'].tolist()

	L = len(dataset)
	ngrams = [defaultdict(float) for _ in range(num_genres)]

''' -----------------------------------------------------------------------------------------'''

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
	@param splitchar: The char at which to split the song lyrics (e.g.: '\n')
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




