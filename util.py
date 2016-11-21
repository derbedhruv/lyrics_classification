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

def generate_homogenous_dataset():
	"""
	Generate a dataset of randomly selected datapoints with equal numbers of each class
	"""

def performance(prediction_function, testdata, class_labels):
	"""
	@param testdata: The testdata in the same form as the training data, i.e. a list of tuples (feature, class)
	"""
	# find the number of rows for each genre type
	genre_count = defaultdict(int)
	for row in testdata:
		genre_count[row[1]] += 1

	# Now find the precision and recall for each genre
	# dicts mapping genres to actual values
	fp = defaultdict(float)
	tp = defaultdict(float)
	tn = defaultdict(float)
	fn = defaultdict(float)

	for row in testdata:
		# get classification for song
		ground_truth_test_genre = row[1]
		ground_truth_test_lyric = row[0]
		predicted_genre = prediction_function(ground_truth_test_lyric)
		# print 'predicted_genre:', predicted_genre, 'genre:', ground_truth_test_genre
		if predicted_genre == ground_truth_test_genre:
			# true positive for this genre
			tp[ground_truth_test_genre] += 1
			# true negative for all other genres
			for g in [x for x in range(len(class_labels)) if x != ground_truth_test_genre]:
				tn[g] += 1
		else:
			# wrong prediction, this is a false negative for this genre
			fn[ground_truth_test_genre] += 1
			# and it is a false positive for all others
			for g in [x for x in range(len(class_labels)) if x != ground_truth_test_genre]:
				fp[g] += 1
	precision = defaultdict(float)
	recall = defaultdict(float)
	accuracy = defaultdict(float)

	def pretty_print(instring):
		"""
		Aligns while printing out by putting appropriate number of tabs
		"""
		print instring,
		tabs = '\t\t'
		if len(instring) > 5:
			 tabs = '\t'
		print tabs,


	print 'Genre\t\tPrecision\tRecall\tF-1 Score'
	for genre_index in range(len(class_labels)):
		try:
			precision[genre_index] = round(tp[genre_index]/(tp[genre_index] + fp[genre_index]), 4)
			recall[genre_index] = round(tp[genre_index]/(tp[genre_index] + fn[genre_index]), 4)

			f1 = round(2*precision[genre_index]*recall[genre_index]/(precision[genre_index] + recall[genre_index]), 4)

			pretty_print(class_labels[genre_index])
			print precision[genre_index], '\t\t', recall[genre_index], '\t\t', f1
		except ZeroDivisionError:
			# happens when tp and fp,fn are 0 due to not enough data being there (hence denominator becomes 0)
			pretty_print(class_labels[genre_index])
			print 'NA\t\tNA\t\tNA'

	print 'Accuracy : ', float(sum(x for x in tp) + sum(x for x in tn))/len(testdata)

# Caluclating performance for baseline
if __name__ == "__main__":
	import classifiers
	# pandas also adds the index of the row, will be removed in this process
	train_data = prepare_data('train.csv')

	# train stochastic gradient descent on this, get weights
	genre_labels = ['Rock', 'Pop', 'Hip Hop/Rap', 'R&B;', 'Electronic', 'Country', 'Jazz', 'Blues', 'Christian', 'Folk']
	baseline = classifiers.Baseline(train_data, genre_labels, debug=True)
	baseline.stochastic_grad_descent()
	baseline.saveModel('baseline-21Nov16.txt')

	# Next, find precision recall for all these
	test_data = prepare_data('test.csv')
	performance(baseline.predict, test_data, genre_labels)




