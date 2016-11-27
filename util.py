""" ----------------------------------------------------------------
	Useful functions that will be referred to in the process of feature extraction and generation of statistics about a song lyric.

---------------------------------------------------------------- """
# extract statistics about a song
import pandas
from collections import defaultdict, OrderedDict
import math
import pickle
import operator
import nltk
from stop_words import get_stop_words	# https://pypi.python.org/pypi/stop-words
import rid 	# https://github.com/jefftriplett/rid.py

stop_words = get_stop_words('en')


# default values for important stuff
### ------------------------------------------------------------------------------------------####
filename = 'songData-Nov26.csv'	# <------- ONLY CHANGE THIS, THE REST ARE DERIVED FROM IT
genres = ['Rock', 'Pop', 'Hip Hop/Rap', 'R&B;', 'Electronic', 'Country', 'Jazz', 'Blues', 'Christian', 'Folk']
### ------------------------------------------------------------------------------------------####

num_genres = len(genres)

def get_genres():
	return genres

def get_filename():
	return filename	

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
def sentence_stats(song_string, ridict, topwords, topngrams, n):
	"""
	@param song_string: string representation of the song lyrics
	@param ridict: RegressiveImageryDictionary object instance
	@param topwords: A list of top words to be bag-of-words type features 
					(i.e. appended to the feature vector as no of occurences of that word in song_string)
	@param topngrams: A list of top ngrams
	@param n: The 'n' of the ngrams
	THE feature generator. Returns a list of ints, corresponding to preset ordered features.
	"""
	# input a string representation of a song's lyrics
	sentences = song_string.split('\n')
	words = song_string.split()

	stats = []
	# important numerical features
	stats.append(song_string.count('\n\n'))
	stats.append(len(sentences) - stats[0] + 1)		# stats[0] <--> no of verses
	stats.append(sum(s.count(' ') + 1 for s in sentences)/float(len(sentences)))
	stats.append(len(words))
	stats.append(sum(len(w) for w in words)/len(words))

	# ADD FEATURES FOR TOP WORDS (BAG OF {selected} WORDS)
	# reduce only to useful words..
	topwords_only = [w for w in words if w in topwords]
	stats += [topwords_only.count(tw) for tw in topwords]

	# TODO: Add the same for top ngrams
	ngrams = [ng for sentence in sentences for ng in extract_ngrams(sentence, n)]	# extracting 2-grams remember!
	stats += [ngrams.count(ng) for ng in topngrams]

	# ADD FEATURES FOR REGRESSIVE IMAGERY DICTIONARY
	rid_features = analyzeRID(song_string, ridict)
	stats += rid_features

	# return stats - which is itself the feature vector now
	return stats

''' the following functions have been used only once - for the preliminary analysis of songs '''
''' -----------------------------------------------------------------------------------------'''

# return the N most commonly used words in a group of songs entered
def NmostComWords(X_train, y_train, N=200):
	"""
	@param N: Top N words will be output
	Does not consider stopwords!
	"""
	# input is a dataframe with 'lyrics' and 'genre' as headers
	# Returns a defaultdict, with avg no of times word appears in each song in the dataset
	lyrics_set = X_train.tolist()
	genres_set = y_train.tolist()

	L = len(X_train)
	words = [defaultdict(float) for _ in range(num_genres)]

	for i, song in enumerate(lyrics_set):
		considered_words = set([s.lower() for s in song.split()]) - set(stop_words)
		for w in considered_words:
			words[genres_set[i]][w] += 1./L
	# sorting will create a new list of tuples
	sorted_list = range(num_genres)		# empty list with num_genres elements needed
	for i in range(num_genres):
		sorted_list[i] = sorted(words[i].items(), key=operator.itemgetter(1))[-N:]	# http://stackoverflow.com/questions/613183/sort-a-python-dictionary-by-value
	return set(x[0] for y in sorted_list for x in y)


def extract_ngrams(sentence, n):
	# returns a list of ngrams ONLY
	return [ngram for ngram in nltk.ngrams(sentence.lower().split(), n)]

def NMostComNgrams(X_train, y_train, n=2, N=100):
	# input X_train and y_train are parts of the same pandas dataframe
	# NOTE: Setting n=1 will NOT return the same thing as NmostComWords() because the latter disregards stopwords
	# while this function does NOT disregard them.
	# Returns a set of n-grams
	# n - the n-gram to consider
	lyrics_set = X_train.tolist()
	genres_set = y_train.tolist()

	L = len(X_train)
	ngrams = [defaultdict(float) for _ in range(num_genres)]

	for i, song in enumerate(lyrics_set):
		for line in song.split('\n'):
			for ngram in nltk.ngrams(line.lower().split(), n):
				ngrams[genres_set[i]][ngram] += 1./L

	# sorting will create a new list of tuples
	sorted_list = range(num_genres)		# empty list with num_genres elements needed
	for i in range(num_genres):
		sorted_list[i] = sorted(ngrams[i].items(), key=operator.itemgetter(1))[-N:]	# http://stackoverflow.com/questions/613183/sort-a-python-dictionary-by-value
	return set(x[0] for y in sorted_list for x in y)

# Setting up Regressive Imagery Dictionary stuff
def setupRID():
	# Defining a new RID
	# taken from https://github.com/jefftriplett/rid.py/blob/master/rid.py
	# The different categories that can be judged are :
	# PRIMARY - 
	##	NEED: ORALITY, ANALITY, SEX 
	##	SENSATION: TOUCH, TASTE, ODOR, GENERAL-SENSATION, SOUND, VISION, COLD, HARD, SOFT
	##	DEFENSIVE SYMBOLIZATION: PASSIVITY, VOYAGE, RANDOM MOVEMENT, DIFFUSION, CHAOS
	##	REGRESSIVE COGNITION: UNKNOWN, TIMELESSNESS, CONSCIOUSNESS ALTERATION, BRINK-PASSAGE, NARCISSISM, CONCRETENESS
	##	ICARIAN IMAGERY: ASCENT, HEIGHT, DESCENT, DEPTH, FIRE, WATER
	# SECONDARY - 
	##	ABSTRACTION, SOCIAL BEHAVIOR, INSTRUMENTAL BEHAVIOR, RESTRAINT, ORDER, TEMPORAL REFERENCES, MORAL IMPERATIVE
	# EMOTIONS - 
	## 	POSITIVE AFFECT, ANXIETY, SADNESS, AFFECTION, AGGRESSION, EXPRESSIVE BEHAVIOR, GLORY
	ridict = rid.RegressiveImageryDictionary()
	ridict.load_dictionary_from_string(rid.DEFAULT_RID_DICTIONARY)
	ridict.load_exclusion_list_from_string(rid.DEFAULT_RID_EXCLUSION_LIST)

	return ridict

def analyzeRID(song_string, ridict):
	# return a vector of results of RID
	RID_CATEGORIES = [
	'PRIMARY:NEED:ORALITY', 'PRIMARY:NEED:ANALITY', 'PRIMARY:NEED:SEX',
	'PRIMARY:SENSATION:TOUCH', 'PRIMARY:SENSATION:TASTE', 'PRIMARY:SENSATION:ODOR', 'PRIMARY:SENSATION:GENERAL-SENSATION', 'PRIMARY:SENSATION:SOUND', 'PRIMARY:SENSATION:VISION', 'PRIMARY:SENSATION:COLD', 'PRIMARY:SENSATION:HARD', 'PRIMARY:SENSATION:SOFT',
	'PRIMARY:DEFENSIVE SYMBOLIZATION:PASSIVITY', 'PRIMARY:DEFENSIVE SYMBOLIZATION:VOYAGE', 'PRIMARY:DEFENSIVE SYMBOLIZATION:RANDOM MOVEMENT', 'PRIMARY:DEFENSIVE SYMBOLIZATION:DIFFUSION', 'PRIMARY:DEFENSIVE SYMBOLIZATION:CHAOS',
	'PRIMARY:REGRESSIVE COGNITION:UNKNOWN', 'PRIMARY:REGRESSIVE COGNITION:TIMELESSNESS', 'PRIMARY:REGRESSIVE COGNITION:CONSCIOUSNESS ALTERATION', 'PRIMARY:REGRESSIVE COGNITION:BRINK-PASSAGE', 'PRIMARY:REGRESSIVE COGNITION:NARCISSISM', 'PRIMARY:REGRESSIVE COGNITION:CONCRETENESS',
	'PRIMARY:ICARIAN IMAGERY:ASCENT', 'PRIMARY:ICARIAN IMAGERY:HEIGHT', 'PRIMARY:ICARIAN IMAGERY:DESCENT', 'PRIMARY:ICARIAN IMAGERY:DEPTH', 'PRIMARY:ICARIAN IMAGERY:FIRE', 'PRIMARY:ICARIAN IMAGERY:WATER',
	'SECONDARY:ABSTRACTION', 'SECONDARY:SOCIAL BEHAVIOR', 'SECONDARY:INSTRUMENTAL BEHAVIOR', 'SECONDARY:RESTRAINT', 'SECONDARY:ORDER', 'SECONDARY:TEMPORAL REFERENCES', 'SECONDARY:MORAL IMPERATIVE',
	'EMOTIONS:POSITIVE AFFECT', 'EMOTIONS:ANXIETY', 'EMOTIONS:SADNESS', 'EMOTIONS:AFFECTION', 'EMOTIONS:AGGRESSION', 'EMOTIONS:EXPRESSIVE BEHAVIOR', 'EMOTIONS:GLORY'
	]
	rid_dictionary = OrderedDict()
	for cat in RID_CATEGORIES:
		rid_dictionary[cat] = 0		# initialize all to 0

	results = ridict.analyze(song_string)
	out_vector = []

	# Need to get the values from 'results' easily
	# referring to the source code
	for (category, count) in results.category_count.items():
		rid_dictionary[category.full_name()] = count

	# return vector of results
	return [rid_dictionary[x] for x in rid_dictionary]


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
	ngs = defaultdict(float)
	for line in song.split('\n'):
		for ngram in nltk.ngrams(line.lower().split(), n):
			ngs[ngram] += 1
	return ngs

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




