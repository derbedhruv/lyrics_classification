# Everything related to the RNNs for the CS221 project
# First will convert the entire dataset into a format similar to the Keras IMDB movie dataset
# REF: https://keras.io/datasets/#imdb-movie-reviews-sentiment-classification
# REF: http://machinelearningmastery.com/sequence-classification-lstm-recurrent-neural-networks-python-keras/
# REF: https://blog.keras.io/using-pre-trained-word-embeddings-in-a-keras-model.html
import numpy as np
np.random.seed(1337)
import pandas
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences

from keras.utils.np_utils import to_categorical
from keras.layers import Dense, Input, Flatten
from keras.layers import Conv1D, MaxPooling1D, Embedding
from keras.models import Model

MAX_NB_WORDS = 20000

dataset = pandas.read_csv("songData-Nov26.csv") 	# TODO: Change this to the training dataset only?
texts = dataset["lyrics"].tolist()
labels = dataset["genre"].tolist()

# Next part is straight from Keras https://blog.keras.io/using-pre-trained-word-embeddings-in-a-keras-model.html
tokenizer = Tokenizer(nb_words=MAX_NB_WORDS)
tokenizer.fit_on_texts(texts)
sequences = tokenizer.texts_to_sequences(texts)

word_index = tokenizer.word_index
print('Found %s unique tokens.' % len(word_index))

data = pad_sequences(sequences, maxlen=MAX_SEQUENCE_LENGTH)

labels = to_categorical(np.asarray(labels))
print('Shape of data tensor:', data.shape)
print('Shape of label tensor:', labels.shape)

# split the data into a training set and a validation set
indices = np.arange(data.shape[0])
np.random.shuffle(indices)
data = data[indices]
labels = labels[indices]
nb_validation_samples = int(VALIDATION_SPLIT * data.shape[0])

x_train = data[:-nb_validation_samples]
y_train = labels[:-nb_validation_samples]
x_val = data[-nb_validation_samples:]
y_val = labels[-nb_validation_samples:]