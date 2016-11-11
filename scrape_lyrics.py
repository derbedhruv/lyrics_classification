# get lyrics from http://www.songlyrics.com/
# eventually, this will only be used to train the dataset on a server. lyrics will not be stored anywhere.
import requests
from bs4 import BeautifulSoup
import string
from collections import defaultdict

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
genres = defaultdict(int)
for alph in string.lowercase:
	artist_count = 0
	songs_total_alph = 0
	uri = 'http://www.songlyrics.com/'+alph
	response = requests.get(uri)
	x  = response.text
	y = BeautifulSoup(x, "html.parser")
	z = y.findAll("li", { "class" : "li_pagination" })
	# find length of z to know the breadth
	zz = str(z[1]).split('\n')
	# Now we can parse this.. need to go from  zz[1] to zz[n-2]
	soup = BeautifulSoup(str(z[1]), 'html.parser')

	artist_links = [alph]
	for p in y.findAll("li", { "class" : "li_pagination" })[0:-1]:
	  current_link = str(p.find('a'))
	  curr = BeautifulSoup(current_link, 'html.parser')
	  link = curr.find('a')
	  if not link == None:
	    # print link['href']
	    artist_links.append(link['href'])
	artist_count += len(artist_links)
	# Now will go to each of these pages in artist links and retrieve songs
	for i,al in enumerate(artist_links):
		song_count = 0
		uri = 'http://www.songlyrics.com/'+al
		response = requests.get(uri)
		x  = response.text
		y = BeautifulSoup(x, "html.parser")
		lyrix = y.findAll("td", { "class" : "td-item" })
		for a in lyrix:
			a = BeautifulSoup(str(a), 'html.parser')
			art_lyr_text = a.td.string
			art_lyr_text = (str(art_lyr_text)).split()
			# print art_lyr_text[0]
			song_count += int(art_lyr_text[0])
			# and now we go down a further level to get the actual songs of each artist
			# go to artist page
			artist_pages = []
			z = y.findAll("table", { "class" : "tracklist" })	# the table of entries of artists
			soup = BeautifulSoup(str(z[0]), 'html.parser')
			for at in soup.findAll('a'):	# loop through all the anchor tags in that table
			  # print 'checking page of artist', at.text,
			  current_artist_url = str(at['href'])		# start exploring this artist
			  artist_pages.append(current_artist_url)
			  # Now we go to this artists page
			  uri = current_artist_url
			  response = requests.get(uri)	# goto the artists page
			  artist_page_html  = response.text
			  artist_page_html_soup = BeautifulSoup(artist_page_html, "html.parser")
			  # on artist page, get Genre
			  artist_songs = artist_page_html_soup.findAll("div", { "class" : "pagetitle" })  # get the title div which has the genres
			  # check if this div is not empty
			  if len(artist_songs) != 0:
			    artist_songs_soup = BeautifulSoup(str(artist_songs[0]), 'html.parser')   # soup banaao
			    genre = str(artist_songs_soup.a.text)		# <- this is the required genre append to hash table
			    if genres[genre] == 0:
			      genres[genre] = 1
			      print 'found new genre', genre
			    else:
			      print 
			  else:
			    print
			# print 'page', i, 'of', alph, 'songs:', song_count
		songs_total_alph += song_count
	print alph, ':', len(artist_links), ', num_songs:',songs_total_alph
print 'total no of artists:', artist_count
