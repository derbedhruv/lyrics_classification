# Read the data in mxm_lyrics_train.txt
# As given on http://labrosa.ee.columbia.edu/millionsong/musixmatch

f = open('mxm_dataset_train.txt', 'r')

words = f.readlines()[17]
words = words.split(',')

print len(words)
