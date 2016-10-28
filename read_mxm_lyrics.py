# Read the data in mxm_lyrics_train.txt
# As given on http://labrosa.ee.columbia.edu/millionsong/musixmatch
import random
from collections import defaultdict

# constants, can be changed for testing
NUM_SONGS = 10000
NUM_ITERS = 10

# hardcode the genre vector
genre_labels = ['Reggae', 'Latin', 'RnB', 'Jazz', 'Metal', 'Pop', 'Punk', 'Country', 'New Age', 'Rap', 'Rock', 'World', 'Blues', 'Electronic', 'Folk']

''' ---------------------------------------------- FUNCTION DEFINITIONS ---------------------------------------------- '''
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

def stochastic_grad_descent(training_set, genres, numIters, eta):
	'''
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
		for i,weight in enumerate(weights):
		     # return 1 if it is the genre corresponding to this weight, else return -1
		     if yy == genre_labels[i]:
		             y = 1
		     else:
		             y = -1
		     # find hinge loss for each genre vector
		     out += max(0, 1 - y*dotProduct(xx, weight))
		return out
         		
	def increment_weight(xx, yy, weights):
	     # use the increment() function to make things convenient
	     for i,weight in enumerate(weights):
	             # return 1 if it is the genre corresponding to this weight, else return -1
	             if yy == genre_labels[i]:
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
		print "i = ",i,", loss = ", lossFunc
	return weights

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

''' ------------------------------------------------------------------------------------------------ '''

f_train = open('mxm_dataset_train.txt', 'r')

for _ in range(17):
	# vaska first 17 header lines
	f_train.readline()
words = f_train.readline()
words = words.split(',')

# read first 50000 songs into sparse vector
print "reading 50,000 songs for training...",
training_songs = []	# list of songs, each represented by a defaultdict
for i in range(18, NUM_SONGS + 18):
	d = defaultdict(int)
	ligne = f_train.readline()
	ligne = ligne.split(',')
	track_id = ligne[0]
	mxm_id = ligne[1]
	for j in range(2,len(ligne)):
		# put rest of the line into the defaultdict
		wordclass = ligne[j]
		w_no, w_count = map(int, wordclass.split(':'))
		d[words[w_no-1]] = w_count	# word index starts from 1!
	# append the (track_id, mxm_id, defaultdict) to the list of songs
	training_songs.append((track_id, mxm_id, d))
print "done!"

print "reading genre classifications ...",
# read in all genre classifications into a dict 
# 15 genres are 'Reggae', 'Latin', 'RnB', 'Jazz', 'Metal', 'Pop', 'Punk', 'Country', 'New Age', 'Rap', 'Rock', 'World', 'Blues', 'Electronic', 'Folk'
genre = {}
f = open('msd_tagtraum_cd2.cls', 'r')
for _ in range(7,280838):	# skip header of 7 lines, go till end of 280838 lines
	data = f.readline().strip()
	data = data.split('\t')
	genre[data[0]] = tuple(x for x in data[1:])		# put the data into the genre dict
f.close()
print "done!"

# Next we use stochastic gradient descent to train a classifier
print "training..."
weights = stochastic_grad_descent(training_songs, genre, NUM_ITERS, 0.01)
print "training complete."

# will test the training loss on the test set
error = 0
count = 0
keyerror = 0
for song in training_songs:
	count += 1
	try:
		if (predict_genre(weights, song[2]) != genre[song[0]][0]):
			error += 1
	except KeyError:
		keyerror += 1

print "training error:", 100*float(error)/(count - keyerror), "%"

# Now we sample the next 50,000 songs in the original file and check the loss on them
print "reading 200,000 songs for training...",
testing_songs = []	# list of songs, each represented by a defaultdict
for i in range(NUM_SONGS + 18, 2*NUM_SONGS + 18):
	d = defaultdict(int)
	ligne = f_train.readline()
	ligne = ligne.split(',')
	track_id = ligne[0]
	mxm_id = ligne[1]
	for j in range(2,len(ligne)):
		# put rest of the line into the defaultdict
		wordclass = ligne[j]
		w_no, w_count = map(int, wordclass.split(':'))
		d[words[w_no-1]] = w_count	# word index starts from 1!
	# append the (track_id, mxm_id, defaultdict) to the list of songs
	testing_songs.append((track_id, mxm_id, d))
print "done! Now will test..."

# testing the model
error = 0
count = 0
keyerror = 0
for song in testing_songs:
	count += 1
	try:
		if (predict_genre(weights, song[2]) != genre[song[0]][0]):
			error += 1
	except KeyError:
		keyerror += 1

print "testing error:", 100*float(error)/(count - keyerror), "%"