# get lyrics from http://www.songlyrics.com/
# eventually, this will only be used to train the dataset on a server. lyrics will not be stored anywhere.
import requests
from bs4 import BeautifulSoup

# de-capitalize names of artist and songs, replace spaces with hyphens
def get_song_lyrics(artist, song):
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
		img = soup.img.extract()
		# convert to text and split at newline
		sentence_case_lyrics = soup.get_text().split('\n')
		for a in sentence_case_lyrics[6:]:
			print a
		# print soup.prettify()
	else:
		print "Could not find song"

# scrape the whole website slowly
uri = 'http://www.songlyrics.com/a/'
response = requests.get(uri)
x  = response.text
y = BeautifulSoup(x, "html.parser")
z = y.findAll("li", { "class" : "li_pagination" })
# find length of z to know the breadth
zz = str(z[1]).split('\n')
# Now we can parse this.. need to go from  zz[1] to zz[n-2]
soup = BeautifulSoup(str(z[1]), 'html.parser')


for p in y.findAll("li", { "class" : "li_pagination" })[1:-1]:
  current_link = str(p.find('a'))
  curr = BeautifulSoup(current_link, 'html.parser')
  link = curr.find('a')
  if not link == None:
    print link['href']
