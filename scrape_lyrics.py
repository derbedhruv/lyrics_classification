# get lyrics from http://www.songlyrics.com/
# eventually, this will only be used to train the dataset on a server. lyrics will not be stored anywhere.
import requests
from bs4 import BeautifulSoup
import string
from collections import defaultdict
import MySQLdb

# global constants
MIN_SONGS = 25	# minimum number of songs for artist to be considered
MIN_LYRICS_CHARCOUNT = 50	# minimum character count for a song's lyrics to be considered

# Now connect to the database - store credentials in ~/.my.cnf
print 'will establish connection to db'
db = MySQLdb.connect(host="localhost", db="cs221_nlp", read_default_file='~/.my.cnf')
db_cursor = db.cursor()

# scrape the whole website slowly

# this will increase the efficiency and speed of artists scraped, since mysql can take HUGE transactions per second
genres = defaultdict(int)
def extract_artist(alph, artist_page = 0, genres_considered=['Rock', 'Pop', 'Hip Hop/Rap', 'R&B;', 'Electronic', 'Country', 'Jazz', 'Blues', 'Christian', 'Folk']):
	"""
	@param alph: single char, should be one of string.lowercase
	@param genres: the list of genres to be considered for scraping (default all 10)
	@param artist_page: the artist pagination to start from (default 0)
	Scrapes songlyrics.com for artist names starting with @alph (should be single letter of english alphabet)
	"""
	assert (alph in string.lowercase), "Only english alphabets allowed as arguments for extract_artist"
	artist_count = 0
	songs_total_alph = 0
	artist_links = [alph]
	# start by expliring each alphabetical URL
	uri_alph = 'http://www.songlyrics.com/'+alph
	response = requests.get(uri_alph)
	html_alph  = response.text
	soup_html_alph = BeautifulSoup(html_alph, "html.parser")
	alph_pagination = soup_html_alph.findAll("li", { "class" : "li_pagination" })
	# Now we can parse this.. need to go from  zz[1] to zz[n-2]
	soup = BeautifulSoup(str(alph_pagination[1]), 'html.parser')

	# now we make a list of all pages for artists starting with 'alph' - then we will iterate through it
	for p in alph_pagination[0:-1]:
	  current_link = str(p.find('a'))
	  curr = BeautifulSoup(current_link, 'html.parser')
	  link = curr.find('a')
	  if not link == None:
	    # print link['href']
	    artist_links.append(link['href'])
	artist_count += len(artist_links)
	# Now will go to each of these pages in artist links and retrieve songs
	for i, al in enumerate(artist_links[artist_page:]):
		song_count = 0
		uri = 'http://www.songlyrics.com/'+al
		response = requests.get(uri)
		x  = response.text
		y = BeautifulSoup(x, "html.parser")
		lyrix = y.findAll("td", { "class" : "td-item" })
		print 'exploring', al
		# and now we go down a further level to get the actual songs of each artist
		artist_pages = []
		z = y.findAll("table", { "class" : "tracklist" })	# the table of entries of artists
		soup = BeautifulSoup(str(z[0]), 'html.parser')
		for at in soup.findAll('tr'):	# loop through all the anchor tags in that table
		  # print 'checking page of artist', at.text,
		  current_artist_url = str(at.a['href'])		# start exploring this artist
		  songs_count_text = at.find("td", {"class": "td-item"}).text
		  num_songs_for_artist = int(songs_count_text.split()[0])
		  artist_name = at.a.text
		  artist_name = artist_name.encode('utf-8')
		  if num_songs_for_artist <= MIN_SONGS:
		    print 'skipping', artist_name, 'due to very few songs'
		  else:
		  	print 'exploring', artist_name
		  artist_pages.append(current_artist_url)
		  # Now we go to this artists page
		  try:
		    # protecting against too many redirects
		    uri = current_artist_url
		    response = requests.get(uri)	# goto the artists page
		  except requests.exceptions.TooManyRedirects:
		    continue	# skip this one
		  artist_page_html  = response.text
		  artist_page_html_soup = BeautifulSoup(artist_page_html, "html.parser")

		  # on artist page, get Genre
		  artist_title = artist_page_html_soup.findAll("div", { "class" : "pagetitle" })  # get the title div which has the genres
		  artist_songs = artist_page_html_soup.findAll("table", { "class" : "tracklist" })	# get all songs
		  songs_seen = 1
		  # check if this div is not empty
		  if len(artist_title) != 0:
		    '''if this is the case, we will be skipping this set of lyrics since they are not tagged'''
		    artist_title_soup = BeautifulSoup(str(artist_title[0]), 'html.parser')   # soup banaao
		    genre = artist_title_soup.a.text		# <- this is the required genre append to hash table
		    genre = genre.encode('utf-8')
		    # Check if this genre is in the list, otherwise skip
		    if not genre in genres_considered:
		    	print 'not considering genre', genre
		    	continue
		    genres[genre] += 1	# add to the new genre list
		    song_count += 1
		    # print current_artist_url, genre
		    # Now finally, get the lyrics and put them into the DB
		    song_soup = soup = BeautifulSoup(str(artist_songs), 'html.parser')
		    for song in song_soup.findAll('a'):
		      if songs_seen <= num_songs_for_artist:
		        # deep-dive into each link one by one and retrieve the lyrics
		        song_url = song['href']
		        song_url = song_url.encode('utf-8')
		        song_name = song.text
		        song_name = song_name.encode('utf-8')
		        song_request = requests.get(song_url)
		        song_lyrics = BeautifulSoup(song_request.text, "html.parser")
		        img = song_lyrics.img.extract()
		        lyrics_div = song_lyrics.find("p", { "id" : "songLyricsDiv" })
		        if not lyrics_div == None:
		          lyric = lyrics_div.get_text()
			  lyric = lyric.encode('utf-8')
		          if len(lyric) >= MIN_LYRICS_CHARCOUNT:
				  '''INSERTING THE LYRICS INTO THE DB'''
				  db_cursor.execute("""insert into song (lyrics, genre, url, artist_name, song_name) values (%s, %s, %s, %s, %s)""", (lyric, genre, song_url, artist_name, song_name))
				  db.commit()
		      songs_seen += 1
		print 'page', i, 'of', alph, 'songs:', song_count
		print dict(genres)
		songs_total_alph += song_count
	print 'RESULTS FOR CASE', alph
	print alph, ':', len(artist_links), ', num_songs:',songs_total_alph
	print 'total no of artists:', artist_count
	# print no of songs for each genre
	print 'total songs per genre:'
	for g in genres:
	  print g, ':', genres[g]

if __name__ == "__main__":
	# alphabets_list = string.lowercase
	alphabets_list = ['p']
	for alph in alphabets_list:
		# TODO: spawn a new thread for each call to extract_artist
		extract_artist(alph, artist_page = 41, genres_considered = ['Christian', 'Folk'])

