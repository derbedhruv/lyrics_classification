# Read the data in mxm_lyrics_train.txt
# As given on http://labrosa.ee.columbia.edu/millionsong/musixmatch
from collections import defaultdict

f = open('mxm_dataset_train.txt', 'r')

words = f.readlines()[17]
words = words.split(',')

# read first 10000 songs into sparse vector
songs = []	# list of songs, each represented by a defaultdict
for i in range(17,10000):
	d = defaultdict(int)
	ligne = f.readline()
	ligne = ligne.split(',')
	track_id = ligne[0]
	mxm_id = ligne[1]
	for j in range(2,len(ligne)):
		# put rest of the line into the defaultdict
		wordclass = ligne[j]
		w_no, w_count = map(int, wordclass.split(':'))
		d[words[w_no-1]] = w_count	# word index starts from 1!
	# append the (track_id, mxm_id, defaultdict) to the list of songs
	songs.append((track_id, mxm_id, d))
f.close()

# read in all genre classifications into a dict 
genre = {}
f = open('msd_tagtraum_cd2.cls', 'r')
for _ in range(7,280838):	# skip header of 7 lines, go till end of 280838 lines
	data = f.readline().strip()
	data = data.split('\t')
	genre[data[0]] = tuple(x for x in data[1:])		# put the data into the genre dict