"""
Useful functions that will be referred to in the process of feature extraction and generation of statistics about a song lyric.
"""
# extract statistics about a song
import pandas
from collections import defaultdict
import nltk
import random
import math
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
	def hinge_loss(xx, yy, weights):
		"""
		@param xx: song features
		@param yy: song label (ground truth)
		@param weights: the weights of the model (dict of dicts)
		Computes the hinge loss in 0-1 prediction of xx as yy
		"""
		out = 0
		for genre_label_index, weight in enumerate(weights):
		     # return 1 if it is the genre corresponding to this weight, else return -1
		     if yy == genre_label_index:
		             y = 1
		     else:
		             y = -1
		     # find hinge loss for each genre vector
		     out += max(0, 1 - y*dotProduct(xx, weight))
		return out
         		
	def increment_weight(xx, yy, weights):
		"""
		@param xx: the song features
		@param yy: the genre (ground truth)
		@param weights: the current model weights to increment
		"""
		# use the increment() function to make things convenient
		for genre_label_index, weight in enumerate(weights):
			# return 1 if it is the genre corresponding to this weight, else return -1
			if yy == genre_label_index:
				y = 1
			else:
				y = -1
			if y*dotProduct(weight, xx) < 1:
				increment(weight, eta*y, xx)

	''' STOCHASTIC GRADIENT DESCENT '''
	for i in range(numIters):
		# calculate loss function with current vector 'weights'
		lossFunc = 0
		for song in training_set:
			# extract lyrics and genre label
			song_lyric = song[0]
			song_genre = song[1]

			# pass these two to the cumulative loss function
			lossFunc += hinge_loss(song_lyric, song_genre, weights)/D

			# choose random vector element and update the gradient for that
			random_song = random.sample(training_set, 1)[0]		# returns a list of single tuple, need to extract that tuple
			random_song_lyric = random_song[0]
			random_song_genre = random_song[1]
			increment_weight(random_song_lyric, random_song_genre, weights)
		# print "i = ",i,", loss = ", lossFunc
	return weights

def predict_genre_sgd(weights, x, genre_labels):
	"""
	@param list weights: A list of dict objects with the weights for each genre
	@param defaultdict x: Feature vector of the song (i.e. occurences of words)
	This function returns the predicted genre as a string, i.e. one of genre_labels
	Which is the one which gives the max dotProduct with the corresonding genre's weight
	"""
	_, highest_weight_label = max((dotProduct(weight, x), i) for i, weight in enumerate(weights))
	return highest_weight_label

def sgd_performance(weights, testdata, genre_labels):
	"""
	@param list weights: A list of dict objects with the weights for each genre
	@param testdata: The testdata in the form
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
		predicted_genre = predict_genre_sgd(weights, ground_truth_test_lyric, genre_labels)
		# print 'predicted_genre:', predicted_genre, 'genre:', ground_truth_test_genre
		if predicted_genre == ground_truth_test_genre:
			# true positive for this genre
			tp[ground_truth_test_genre] += 1
			# true negative for all other genres
			for g in [x for x in range(len(genre_labels)) if x != ground_truth_test_genre]:
				tn[g] += 1
		else:
			# wrong prediction, this is a false negative for this genre
			fn[ground_truth_test_genre] += 1
			# and it is a false positive for all others
			for g in [x for x in range(len(genre_labels)) if x != ground_truth_test_genre]:
				fp[g] += 1
	precision = defaultdict(float)
	recall = defaultdict(float)
	accuracy = defaultdict(float)

	print 'Genre\t\tPrecision\tRecall\tF-1 Score'
	for genre_index in range(len(genre_labels)):
		try:
			precision[genre_index] = tp[genre_index]/(tp[genre_index] + fp[genre_index])
			recall[genre_index] = tp[genre_index]/(tp[genre_index] + fn[genre_index])
			f1 = 2*precision[genre_index]*recall[genre_index]/(precision[genre_index] + recall[genre_index])
			print genre_labels[genre_index], '\t\t', precision[genre_index], '\t\t', recall[genre_index], '\t\t', f1
		except ZeroDivisionError:
			# happens when tp and fp,fn are 0 due to not enough data being there (hence denominator becomes 0)
			print genre_labels[genre_index], '\t\tNA\t\tNA\t\tNA'
	print len(testdata)
	print 'Accuracy : ', (sum(x for x in tp) + sum(x for x in tn))/len(testdata)


# implementing a random forest classifier
# TODO: define as a class
class RandomForestClassifier():
	# A class of problems which are the random forest classifier
	# trains a classifier given the data, number of trees and number of features

	def __init__(self, training_data, num_trees, num_features, class_labels):
		"""
		@param dataset: A dataset of points (x, y) where x is a feature vector 
						(defaultdict) representation of a song and y is the 
						genre (0...9) that song belongs to
		@param num_features: the number of features to (randomly) consider for 
						finding the best split. More features takes longer but
						would likely give better results
		@param class_labels: The set of all possible output class labels
		"""
		self.num_features = num_features
		self.num_trees = num_trees
		self.data = training_data
		self.class_labels = class_labels


	def leaf_node(self, group):
		"""
		@param group: A group (subset) of data as a list of (features_dict, class)
		Create a leaf node class, as the mode of the classes in the group
		"""
		classes = [x[1] for x in group]
		return max(classes, key=classes.count)

	def split_into_groups(self, rfeature, value):
		"""
		@param dataset: A dataset of points (x, y) where x is a feature vector 
						(defaultdict) representation of a song and y is the 
						genre (0...9) that song belongs to
		@param rfeature: One selected feature
		@param value: One value of rfeature

		Splits the dataset into two groups, one with value at 'rfeature' greater
		or lesser than 'value'. Returns both groups
		"""
		group_lower = []
		group_higher = []
		for datapoint in self.data:
			if datapoint[0][rfeature] > value:
				group_higher.append(datapoint)
			else:
				group_lower.append(datapoint)
		return (group_lower, group_higher)

	def gini_impurity(self, groups):
		"""
		Gini impurity is a measure of how often a randomly chosen element 
		from the set would be incorrectly labeled if it was randomly labeled 
		according to the distribution of labels in the subset
		https://en.wikipedia.org/wiki/Decision_tree_learning#Gini_impurity
		"""
		gini_impurity = 0.
		for label in self.class_labels:
			for group in groups:
				group_size = float(len(group))	# float so quotient is also float
				if group_size == 0:
					# there's no impurity that can be measured 
					continue
				group_labels = [x[1] for x in group]	# each element of group is (features, label)
				label_ratio = group_labels.count(label)/group_size
				gini_impurity += label_ratio*(1 - label_ratio)
		return gini_impurity


	def find_best_split(self):
		"""
		Returns the best split point, i.e. a feature and corresponding value 
		of one feature out of 'num_features' candidates which splits 'dataset' into the 
		most homogenous distribution of classes (genres)
		"""
		groups = None
		best_split_feature = None
		best_split_score = 10000	# arbitrary large number
		best_split_feature_value = 10000
		# first, randomly select unique num_features out of all features
		all_features = set(y for t in self.data for y in t[0].keys())
		random_feature_set = random.sample(all_features, self.num_features)

		# for each of these randomly sampled features, go through all 
		# rows in dataset and find the gini index of each split
		for rfeature in random_feature_set:
			for datapoint in self.data:
				split_groups = self.split_into_groups(rfeature, datapoint[0][rfeature])
				split_purity = self.gini_impurity(split_groups)
				if split_purity < best_split_score:
					groups = split_groups
					best_split_score = split_purity
					best_split_feature = rfeature
					best_split_feature_value = datapoint[0][rfeature]
		return {'groups': groups, 'best_split_feature':best_split_feature, 'best_split_score':best_split_score, 'best_split_feature_value':best_split_feature_value}

	def generate(self):
		# fit the classifier
		return self.find_best_split()	# THIS IS ONLY FOR


# read in train and test data - to make it easy to debug on python terminal
def prepare_data(filename):
	train_data = pandas.read_csv(filename)
	train_data = train_data.to_records(index=False)		# Now is a list of tuples (lyrics, genre)
	train_data = [(bag_of_words(l), g) for i,l,g in train_data]	
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




