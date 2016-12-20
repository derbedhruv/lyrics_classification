# the flask app for the baseline for the cs221 mini project
import os, util, sys, pickle, re
from flask import Flask, request, Response, jsonify

import rid
import util
from sklearn.externals import joblib	# better than pickle

import numpy
from sklearn.manifold import TSNE



app_folder = os.path.dirname(os.path.realpath(__file__))
app = Flask(__name__)

@app.route('/')
def index():
  return Response(open(os.path.abspath(os.path.join(app_folder, '..', '/index.html'))).read(), mimetype="text/html")


@app.route('/cs221_project')
def cs221():
  return Response(open(app_folder + '/index.html').read(), mimetype="text/html")

@app.route('/process-lyrics', methods=['GET', 'POST'])
def process_lyrics():
	# Receive the lyrics, process them and send back vector, final prediction, other interesting information
	# print "LYRICS:", request.form['lyrics']
	lyrics = request.form['lyrics']		# the string form of the lyrics input

	# send data containing genre probability matrix, RID output and sentence stats?
	# At some point in the future can output the vector space representation of each word
	processed = {}

	# RID processing
	ridict = util.setupRID()	# create new RID
	processed['rid'] = [(category.full_name(), count) for (category, count) in ridict.analyze(lyrics).category_count.items()]
	print processed['rid']

	# convert to vector
	# load the topwords and topngrams
	topwords = pickle.load(open(app_folder +'/flaskapp-topwords.pklz', 'r'))
	topngrams = pickle.load(open(app_folder + '/flaskapp-topngrams.pklz', 'r'))

	# loading the GloVe dictionary
	glove_dictionary = {}
	f = open(app_folder + '/glove.custom.70percentLyricsCorpus.Dec10.txt')     # custom made GloVe representation capturing the semantics of lyrics corpus
	for line in f:
	    values = line.split()
	    word = values[0]
	    coefs = numpy.asarray(values[1:], dtype='float32')
	    glove_dictionary[word] = coefs

	f.close()

	# then will find which words are being used in this case
	wordpoints = []		# will be converted in a JSON containing an array of (word, x, y, z)
	wordarray = []
	words = [re.sub(r"[^\s\w_]+", '', w.lower()) for w in lyrics.split()]
	for word in words:
		wordarray.append(list(glove_dictionary[word]))

	# lower dimensions and fit..
	if len(wordarray) > 1:
		# can only run TSNE if there's more than one word!
		tsne = TSNE(n_components=3, random_state=0)
		P = tsne.fit_transform(wordarray)

		for i, word in enumerate(words):
			tempwpoint = {}
			tempwpoint['word'] = word
			tempwpoint['x'] = P[i][0]
			tempwpoint['y'] = P[i][1]
			tempwpoint['z'] = P[i][2]

			# finally append it to the main dict
			wordpoints.append(tempwpoint)

	# scan through the string and find out which of the top words and n-grams are present
	topwords_present = [w for w in lyrics.split() if w in topwords]
	topngrams_present = [ng for ng in util.ngram(lyrics, n=3) if ng in topngrams]

	# convert the string to vector
	song_vector = util.sentence_stats(lyrics, ridict, topwords, topngrams, n=3)

	# load the model and make prediction
	# http://stackoverflow.com/questions/23000693/how-to-output-randomforest-classifier-from-python
	RFC = joblib.load(app_folder + '/rfc-cs221-poster.pklz')
	# prediction = RFC.predict_proba(song_vector.reshape(1, -1))[0].tolist()
	prediction = RFC.predict_proba(song_vector)[0].tolist()

	# append to the dictionary
	processed['probabilities'] = prediction
	processed['stats'] = song_vector[:5]	# send the first 5 of the features - these are the sentence stats
	processed['topwords'] = topwords_present
	processed['topngrams'] = topngrams_present
	processed['prediction'] = RFC.predict(song_vector).tolist()[0]
	processed['wordpoints'] = wordpoints

	# return json version of dictionary to requester
	# http://stackoverflow.com/questions/13081532/how-to-return-json-using-flask-web-framework
	return jsonify(**processed)

if __name__ == "__main__":
  app.run()
