''' ----------------------------------------------------------------
	ALL CLASSIFIERS WHICH HAVE BEEN WRITTEN BY ME

	ML classifiers have been written as classes for ease of logic
	They make use of common functions in util.py
---------------------------------------------------------------- '''
from util import *
import random
import math
import pickle

# baseline class
class Baseline():
	# A logistic regression classifier which uses stochastic gradient descent on hinge loss

	def __init__(self, training_data, class_labels, numIters=10, eta=0.01, debug=False):
		"""
		@param training_set: List of (song, genre)
		@param genres: list of genres used
		@param numIters: number of iterations to run, 10 is usually more than enough
		@param eta: step size for stochastic gradient descent, can play around to find optimal
		"""
		self.training_set = training_data
		self.class_labels = class_labels
		# weights is a list of dicts, one for each class label
		self.weights = weights = [{} for _ in range(len(self.class_labels))]
		self.numIters = numIters
		self.eta = eta
		self.debug = debug

	def stochastic_grad_descent(self):
		'''
		Given training_set, which is a list of (track_id, mxm_id, vector) tuples. 
		The 'vector' is a sparse vector containing number of times a word occurs
		in a song's lyrics.
		This function will return the weight vector (sparse
		feature vector) learned using stochastic gradient descent.
		'''
		D = len(self.training_set)
		random.seed(88)
		def hinge_loss(xx, yy):
			"""
			@param xx: song features
			@param yy: song label (ground truth)
			Computes the hinge loss in 0-1 prediction of xx as yy
			"""
			out = 0
			for genre_label_index, weight in enumerate(self.weights):
			     # return 1 if it is the genre corresponding to this weight, else return -1
			     if yy == genre_label_index:
			             y = 1
			     else:
			             y = -1
			     # find hinge loss for each genre vector
			     out += max(0, 1 - y*dotProduct(xx, weight))
			return out
	         		
		def increment_weight(xx, yy):
			"""
			@param xx: the song features
			@param yy: the genre (ground truth)
			@param weights: the current model weights to increment
			"""
			# use the increment() function to make things convenient
			for genre_label_index, weight in enumerate(self.weights):
				# return 1 if it is the genre corresponding to this weight, else return -1
				if yy == genre_label_index:
					y = 1
				else:
					y = -1
				if y*dotProduct(weight, xx) < 1:
					increment(weight, self.eta*y, xx)

		''' STOCHASTIC GRADIENT DESCENT '''
		for i in range(self.numIters):
			# calculate loss function with current vector 'weights'
			lossFunc = 0
			for song in self.training_set:
				# extract lyrics and genre label
				song_lyric = song[0]
				song_genre = song[1]

				# pass these two to the cumulative loss function
				lossFunc += hinge_loss(song_lyric, song_genre)/D

				# choose random vector element and update the gradient for that
				random_song = random.sample(self.training_set, 1)[0]		# returns a list of single tuple, need to extract that tuple
				random_song_lyric = random_song[0]
				random_song_genre = random_song[1]
				increment_weight(random_song_lyric, random_song_genre)
			if self.debug == True:
				print "iteration = ", i, ", loss = ", lossFunc

	def predict(self, x):
		"""
		@param defaultdict x: Feature vector of the song (i.e. occurences of words)
		This function returns the predicted genre as a string, i.e. one of genre_labels
		Which is the one which gives the max dotProduct with the corresonding genre's weight
		"""
		_, highest_weight_label = max((dotProduct(weight, x), i) for i, weight in enumerate(self.weights))
		return highest_weight_label

	def save(self, filename):
		f = open(filename, 'w')
		pickle.dump(self.weights, f)
		f.close()


# implementing a random forest classifier
class RandomForestClassifier():
	# A class of problems which are the random forest classifier
	# trains a classifier given the data, number of trees and number of features
	# REF: https://www.stat.berkeley.edu/~breiman/RandomForests/cc_home.htm

	def __init__(self, training_data, num_trees, num_features, class_labels, tree_depth, random_seed=None):
		"""
		@param dataset: A dataset of points (x, y) where x is a feature vector 
						(defaultdict) representation of a song and y is the 
						genre (0...9) that song belongs to
		@param num_features: the number of features to (randomly) consider for 
						finding the best split. More features takes longer but
						would likely give better results
		@param class_labels: The set of all possible output class labels
		@param tree_depth: The max depth of each decision tree which would be formed
		@param random_seed: A random seed value, in case of debugging where reproducibility is desired
		"""
		self.num_features = num_features
		self.num_trees = num_trees
		self.data = training_data
		self.class_labels = class_labels
		self.maxdepth = tree_depth

		if not random_seed == None:
			# in case reproducible results are desired
			random.seed(random_seed)

	def create_leaf(self, group):
		"""
		@param group: A group (subset) of data as a list of (features_dict, class)
		Create a leaf node class, as the mode of the classes in the group
		"""
		classes = [x[1] for x in group]
		value = max(classes, key=classes.count)
		# print value
		return value

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


	def find_best_split(self, data):
		"""
		@param data: A subset of the dataset, which will be recursively processed
		Returns the best split point, i.e. a feature and corresponding value 
		of one feature out of 'num_features' candidates which splits 'dataset' into the 
		most homogenous distribution of classes (genres)
		"""
		groups = None
		best_split_feature = None
		best_split_score = 10000	# arbitrary large number
		best_split_feature_value = 10000
		# first, randomly select unique num_features out of all features
		all_features = set(y for t in data for y in t[0].keys())
		# sample num_features features from all_features WITH REPLACEMENT!!
		random_feature_set = [random.sample(all_features, 1)[0] for i in self.num_features]

		# for each of these randomly sampled features, go through all 
		# rows in dataset and find the gini index of each split
		for rfeature in random_feature_set:
			for datapoint in data:
				split_groups = self.split_into_groups(rfeature, datapoint[0][rfeature])
				split_purity = self.gini_impurity(split_groups)
				if split_purity < best_split_score:
					groups = split_groups
					best_split_score = split_purity
					best_split_feature = rfeature
					best_split_feature_value = datapoint[0][rfeature]
		#return {'groups': groups, 'best_split_feature':best_split_feature, 'best_split_score':best_split_score, 'best_split_feature_value':best_split_feature_value}
		return {'groups': groups, 'f':best_split_feature, 'val':best_split_feature_value}

	def get_split(self, node, current_depth):
		# recursive function
		left, right = node['groups']	# extract the two split groups
		del node['groups']	# delete this to prevent excessive memory use
		# base case: check if either of these groups is empty
		if not left or not right:
			# this means that the node entered is a leaf node. 
			# Hence both the left and right values of the node are equal to the majority class
			node['left'] = self.create_leaf(left + right)
			node['right'] = node['left']
			return
		# Check if max depth has been reached
		if current_depth >= self.maxdepth:
			# make both child nodes into leaves
			node['left'] = self.create_leaf(left)
			node['right'] = self.create_leaf(right)
			return
		# Process recursively, if none of these is the case
		node['left'] = self.find_best_split(left)
		self.get_split(node['left'], current_depth+1)
		node['right'] = self.find_best_split(right)
		self.get_split(node['right'], current_depth+1)

	def generate_decision_tree(self):
		# recursively generates a decision tree
		# this will be a defaultdict with 'left' and 'right' keys indicating 
		root_node = self.find_best_split(self.data)
		self.get_split(root_node, current_depth=1)
		return root_node

	def predict(self, node, x):
		"""
		@param root_node: A node of the tree
		@param x: A feature representation of data
		Make a prediction with a decision tree. Recursive function.
		Output is a class label
		"""
		# check if the value of the feature of x given in node 'f' is greater than 'val'
		feature = node['f']
		val = node['val']
		if x[feature] >= val:
			# go to the right node
			# base case: check if node is a leaf node, return val
			if not type(node['right']) == type({}):
				# node is not a dict, hence return its value
				return node['right']
			else:
				return self.predict(node['right'], x)
		else:
			if not type(node['left']) == type({}):
				# node is not a dict, hence return its value
				return node['left']
			else:
				return self.predict(node['left'], x)







