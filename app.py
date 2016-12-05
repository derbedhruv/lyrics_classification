# the flask app for the baseline for the cs221 mini project
from flask import Flask, request, Response, jsonify

import util
from sklearn.externals import joblib	# better than pickle
from sklearn.ensemble import RandomForestClassifier # use predict_proba to output the probabilities of each genre
import rid

import pickle

app = Flask(__name__)

@app.route('/')
def index():
  return Response(open('index.html').read(), mimetype="text/html")

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
	topwords = pickle.load(open('flaskapp-topwords.pklz', 'r'))
	topngrams = pickle.load(open('flaskapp-topngrams.pklz', 'r'))

	# scan through the string and find out which of the top words and n-grams are present
	topwords_present = [w for w in lyrics.split() if w in topwords]
	topngrams_present = [ng for ng in util.ngram(lyrics, n=3) if ng in topngrams]

	# convert the string to vector
	song_vector = util.sentence_stats(lyrics, ridict, topwords, topngrams, n=3)

	# load the model and make prediction
	# http://stackoverflow.com/questions/23000693/how-to-output-randomforest-classifier-from-python
	RFC = joblib.load('rfc-cs221-poster.pklz')
	# prediction = RFC.predict_proba(song_vector.reshape(1, -1))[0].tolist()
	prediction = RFC.predict_proba(song_vector)[0].tolist()

	# append to the dictionary
	processed['classification'] = prediction
	processed['stats'] = song_vector[:5]	# send the first 5 of the features - these are the sentence stats
	processed['topwords'] = topwords_present
	processed['topngrams'] = topngrams_present

	# return json version of dictionary to requester
	# http://stackoverflow.com/questions/13081532/how-to-return-json-using-flask-web-framework
	return jsonify(**processed)

if __name__ == "__main__":
  app.run()