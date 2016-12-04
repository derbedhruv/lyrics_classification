# the flask app for the baseline for the cs221 mini project
from flask import Flask, request, Response

app = Flask(__name__)

@app.route('/')
def index():
  return Response(open('index.html').read(), mimetype="text/html")

@app.route('/process-lyrics', methods=['GET', 'POST'])
def process_lyrics():
	# lyrics = request.form['lyrix']
	print "LYRICS:", request.form['lyrics']
	return 'OK'

if __name__ == "__main__":
  app.run()