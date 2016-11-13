# Read the data in mxm_lyrics_train.txt
# As given on http://labrosa.ee.columbia.edu/millionsong/musixmatch
import random
from collections import defaultdict

# hardcode the genre vector
genre_labels = ['Reggae', 'Latin', 'RnB', 'Jazz', 'Metal', 'Pop', 'Punk', 'Country', 'New Age', 'Rap', 'Rock', 'World', 'Blues', 'Electronic', 'Folk']

''' ---------------------------------------------- FUNCTION DEFINITIONS ---------------------------------------------- '''
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

def read_data_BoW():
	"""
	Read in files from the dataset and put them into a (training_set, testing_set) tuple
	Split is about 80/20
	Each of them is a dict of (bag_of_words, genre), where bag_of_words is a dict of words and no of occurences of the word
	genre is a string with the genre into which the song is classified
	"""
	f_train = open('mxm_dataset_train.txt', 'r')

	for _ in range(17):
	  # vaska first 17 header lines
	  f_train.readline()
	words = f_train.readline()
	words = words.split(',')

	# read songs into sparse vector
	print "reading songs for training...",
	training_songs = []	# list of songs, each represented by a defaultdict
	for i in range(18, 210537):
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
	f_train.close()
	print "done!",
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
	print "done!",

	# Now combine both into a training set
	training_set = []
	for song in training_songs:
		try:
			for g in genre[song[0]]:
				training_set.append((song[2], g))	# take the first ([0]'th) genre of the genres that the current song identifies with
		except KeyError:
			continue

	print "reading songs for training...",
	f_test = open('mxm_dataset_test.txt', 'r')

	for _ in range(17):
	  # vaska first 18 header lines
	  f_test.readline()
	words = f_test.readline()
	words = words.split(',')

	testing_songs = []	# list of songs, each represented by a defaultdict
	for i in range(18, 27161):
	  d = defaultdict(int)
	  ligne = f_test.readline()
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
	f_test.close()
	print "done!"
	# Now combine both into a training set
	testing_set = []
	for song in testing_songs:
		try:
			for g in genre[song[0]]:
				testing_set.append((song[2], g))	# take the first ([0]'th) genre of the genres that the current song identifies with
		except KeyError:
			continue
	return (training_set, testing_set)

''' ------------------------------------------------------------------------------------------------ '''

# Now will use nltk to train a model
