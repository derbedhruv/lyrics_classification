# get lyrics from http://www.songlyrics.com/
# eventually, this will only be used to train the dataset on a server. lyrics will not be stored anywhere.
import requests
from bs4 import BeautifulSoup

# de-capitalize names of artist and songs, replace spaces with hyphens
artist_hyphenated = 'aerosmith' 
song_name = 'remember'

uri = 'http://www.songlyrics.com/' + artist_hyphenated + '/' + song_name + '-lyrics/'
response = requests.get(uri)

if response.status_code == 200:
	x  = response.text
	y = BeautifulSoup(x, "html.parser")
	z = y.findAll("p", { "id" : "songLyricsDiv" })
	# re-parse as html
	soup = BeautifulSoup(str(z[0]), 'html.parser')
	# remove the annoying ad <img> tags
	soup.img.extract()
	# will attempt to remove <br> tags
	print soup
	# print soup.prettify()
else:
	print "Could not find song"
