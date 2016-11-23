# Cleaning the data which has been scraped from songlyrics.com
import sys
import MySQLdb
import re
import pandas as pd
import collections

genres_list = ['Rock', 'Pop', 'Hip Hop/Rap', 'R&B;', 'Electronic', 'Country', 'Jazz', 'Blues', 'Christian', 'Folk']
accepted_characters = set('0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ\',.\n\r ')
SONG_LIMIT_PER_GENRE = 10000	# not taking more than these per genre to have a homogenous mix of training data

# genre, url, lyrics
def get_songs_by_genre(genre_of_interest, db_cursor):
	"""
	@param: genre_of_interest: The genre you are interested in, as a string
	Send query to db for a particular genre
	"""
	query = "select lyrics from song where genre = '%s'" %genre_of_interest
	db_cursor.execute(query)
	data = db_cursor.fetchall()
	return data

def get_data(genres=genres_list):
	"""
	@param genres: List of genres (strings) to collect from the db
	Gets data from the db, arranges it in the form ('lyrics', genre_class), where genre_class is an int representing the genre
	It correspondds to the index in the genres list. Returns a Panda object (dataframe).
	"""
	print 'establishing connection to db...',
	db = MySQLdb.connect(host="localhost", db="cs221_nlp", read_default_file='~/.my.cnf')
	db_cursor = db.cursor()
	print 'done!'
	dataset = []
	for label, genre in enumerate(genres):
		data = get_songs_by_genre(genre, db_cursor)
		for song in data:
			# song is a singleton tuple (since the db query returns only tuples), so need to extract
			dataset.append([song[0], label])
	# convert to pandas
	dataset = pd.DataFrame.from_records(dataset, columns=['lyrics', 'genre'])
	return dataset


if __name__ == "__main__":
	# read in the songs, straight from the db
	data = get_data()

	# convert them into lists
	all_lyrics = data['lyrics'].tolist()
	all_lyrics_genres = data['genre'].tolist()

	# process them: one by one go through each, and process the lyrics. Then append to the corresponding genre list
	songs_by_genre = collections.defaultdict(int)
	songs_master_list = []

	for i in range(len(all_lyrics)):
		if i%1000 == 0:
			print 'Finished processing song #%d' %i
		current_genre = all_lyrics_genres[i]
		if (songs_by_genre[current_genre] > SONG_LIMIT_PER_GENRE):
			continue
		current_song = all_lyrics[i]
		# filter using knowledge from http://stackoverflow.com/questions/8689795/how-can-i-remove-non-ascii-characters-but-leave-periods-and-spaces-using-python
		current_song = filter(lambda x: x in accepted_characters, current_song)		# get rid of characters which aren't in our guest list
		if not (current_song, current_genre) in songs_master_list:
			# expensive operation but worth it to prevent duplicates
			songs_master_list.append((current_song, current_genre))
			songs_by_genre[current_genre] += 1

	print 'Completed. Number of distinct songs = %d' %len(songs_master_list)
	# Now we have a list of lists containing filtered strings. We need to save them in a form that is easily readable. 
	# Pandas are a good choice
	print 'Saving songs to csv file...',
	final_songs = pd.DataFrame(songs_master_list)
	f = open('songData-Nov22.csv', 'w')
	final_songs.to_csv(f)
	f.close()
	print 'completed! Enjoy your new dataset.'


