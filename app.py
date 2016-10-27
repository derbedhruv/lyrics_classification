# the flask app for the baseline for the cs221 mini project
from flask import Flask, request, Response

app = Flask(__name__)

@app.route('/', methods=['GET'])
def index():
  return Response(open('index.html').read(), mimetype="text/html")

if __name__ == "__main__":
  app.run()
