# Everything related to the RNNs for the CS221 project
# First will convert the entire dataset into a format similar to the Keras IMDB movie dataset
# REF: https://keras.io/datasets/#imdb-movie-reviews-sentiment-classification
# REF: http://machinelearningmastery.com/sequence-classification-lstm-recurrent-neural-networks-python-keras/
# REF: https://blog.keras.io/using-pre-trained-word-embeddings-in-a-keras-model.html
import os, re
import datetime
import pickle

import numpy as np
np.random.seed(1337)
import matplotlib
# Force matplotlib to not use any Xwindows backend.
matplotlib.use('Agg')   # http://stackoverflow.com/questions/29217543/why-does-this-solve-the-no-display-environment-issue-with-matplotlib
import matplotlib.pyplot as plt

import pandas
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences

from keras.utils.np_utils import to_categorical
from keras.layers import Dense, Input, Flatten
from keras.layers import Conv1D, MaxPooling1D, Embedding
from keras.models import Model

from sklearn.metrics import classification_report

# ------------------------------#
#  HYPERPARAMETERS TO CONTROL!
# ------------------------------#
MAX_SEQUENCE_LENGTH = 1000      # maximum number of words in a sequence
MAX_NB_WORDS = 10000            # top MAX_NB_WORDS most common words will be used
EMBEDDING_DIM = 100
VALIDATION_SPLIT = 0.2
NUM_EPOCHS = 4
# ------------------------------#

# prepare text samples and their labels
print('Processing text dataset')
# labels_index = {'Jazz': 6, 'Christian': 8, 'Hip Hop/Rap': 2, 'R&B;': 3, 'Rock': 0, 'Pop': 1, 'Country': 5, 'Blues': 7, 'Electronic': 4, 'Folk': 9}	# dictionary mapping label name to numeric id
labels_index = {'Jazz': 5, 'Christian': 7, 'Hip Hop/Rap': 2, 'R&B;': 3, 'Rock': 0, 'Pop': 1, 'Country': 4, 'Blues': 6}	# dictionary mapping label name to numeric id
dataset = pandas.read_csv("songData-Dec3.csv") 	# TODO: Change this to the training dataset only?
texts = dataset["lyrics"].tolist()
labels = dataset["genre"].tolist()

# remove useless characters and convert to lowercase
print "processing text to remove useless chars..."
texts = [re.sub(r"[^\s\w_]+", '', t.lower().replace('\n', ' ')) for t in texts]
print "done!"

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

# ----------------------------------------------------- #
# USE THIS TO SHORTEN THE DATASET FOR QUICK EXPERIMENTS #
# ----------------------------------------------------- #
'''
x_train = x_train[:1000]
y_train = y_train[:1000]
x_val = x_val[:50]
y_val = y_val[:50]
'''
# ----------------------------------------------------- #

# ----------------------------------------------------- #
# GloVe STUFF
# ----------------------------------------------------- #
print('Indexing word vectors. NOTE: You should have the GloVe embeddings available locally. If not, download and unzip from http://nlp.stanford.edu/data/glove.6B.zip')

embeddings_index = {}
f = open('glove.6B.100d.txt')
for line in f:
    values = line.split()
    word = values[0]
    coefs = np.asarray(values[1:], dtype='float32')
    embeddings_index[word] = coefs

f.close()

print('Found %s word vectors.' % len(embeddings_index))

# prepare embedding matrix
print('Preparing embedding matrix.')

nb_words = min(MAX_NB_WORDS, len(word_index))
embedding_matrix = np.zeros((nb_words + 1, EMBEDDING_DIM))
for word, i in word_index.items():
    if i > MAX_NB_WORDS:
        continue
    embedding_vector = embeddings_index.get(word)
    if embedding_vector is not None:
        # words not found in embedding index will be all-zeros.
        embedding_matrix[i] = embedding_vector

# load pre-trained word embeddings into an Embedding layer
# note that we set trainable = False so as to keep the embeddings fixed
embedding_layer = Embedding(nb_words + 1,
                            EMBEDDING_DIM,
                            weights=[embedding_matrix],
                            input_length=MAX_SEQUENCE_LENGTH,
                            trainable=False)
# ----------------------------------------------------- #


print('Training model.')
# ----------------------------------------------------- #
# TRAINING A CONV NEURAL NET
# https://blog.keras.io/using-pre-trained-word-embeddings-in-a-keras-model.html
# ----------------------------------------------------- #
sequence_input = Input(shape=(MAX_SEQUENCE_LENGTH,), dtype='int32')
embedded_sequences = embedding_layer(sequence_input)

# train a 1D convnet with global maxpooling
x = Conv1D(32, 5, activation='relu')(embedded_sequences)
x = MaxPooling1D(5)(x)
x = Conv1D(128, 5, activation='relu')(x)
x = MaxPooling1D(5)(x)
x = Conv1D(128, 5, activation='relu')(x)
x = MaxPooling1D(35)(x)
x = Flatten()(x)
x = Dense(128, activation='relu')(x)
preds = Dense(len(labels_index), activation='softmax')(x)

model = Model(sequence_input, preds)
model.compile(
    loss='categorical_crossentropy',
    optimizer='rmsprop',
    metrics=['acc']
)

# ----------------------------------------------------- #

# ----------------------------------------------------- #
# TRAINING AN LSTM RECURRENT NEURAL NET
# http://machinelearningmastery.com/sequence-classification-lstm-recurrent-neural-networks-python-keras/
# ----------------------------------------------------- #
'''
from keras.layers import LSTM

x = LSTM(100)(embedded_sequences)
x = Dense(1, activation='sigmoid')(x)
preds = Dense(len(labels_index), activation='softmax')(x)

model = Model(sequence_input, preds)
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['acc'])

# print(model.summary())
# model.fit(X_train, y_train, nb_epoch=3, batch_size=64)   # <------- TRY VARYING BATCH SIZE?
'''
# ----------------------------------------------------- #


# actuall fitting..
# Will get history information http://machinelearningmastery.com/display-deep-learning-model-training-history-in-keras/
history = model.fit(x_train, y_train, validation_data=(x_val, y_val), nb_epoch=NUM_EPOCHS, batch_size=128)

# list all data in history
print(history.history.keys())
pickle.dump(history.history['acc'], open("history_acc.pklz", "w"))
pickle.dump(history.history['loss'], open("history_loss.pklz", "w"))

# summarize history for accuracy
plt.figure(0)   # to seperate figures
plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')

# ----------------------------------------------------- #
# SAVE FIGURE BASED ON DATE TIME FOR RECORDS
# ----------------------------------------------------- #
now = datetime.datetime.now()
figname = "accuracy_plot_" + str(now.year) + str(now.month) + str(now.day) + "_" + str(now.hour) + str(now.minute) + "hrs.png"
plt.savefig(figname)
# ----------------------------------------------------- #

# summarize history for loss
plt.figure(1)   # to seperate figures
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
# ----------------------------------------------------- #
# SAVE FIGURE BASED ON DATE TIME FOR RECORDS
# ----------------------------------------------------- #
figname = "loss_history_plot_" + str(now.year) + str(now.month) + str(now.day) + "_" + str(now.hour) + str(now.minute) + "hrs.png"
plt.savefig(figname)
# ----------------------------------------------------- #

# Final evaluation of the model
scores = model.evaluate(x_val, y_val, verbose=0)
print("Accuracy: %.2f%%" % (scores[1]*100))
print "model precision and recall analysis for each genre->"

# calculate the prediction on the test set
y_pred = model.predict(x_val)
# will round the values to 0 or 1 with threshold 0.5 --- TODO: select better threshold using ROC curve?
# y_pred = np.matrix.round(y_pred) 
y_pred = np.argmax(y_pred, axis=1)
y_true = np.argmax(y_val, axis=1)
print classification_report(y_true, y_pred)
