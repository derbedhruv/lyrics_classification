# Lyrics-based classification of songs 
## Stanford CS 221 Fall 2016 Mini Project
Author: Dhruv Joshi

The purpose of this course mini-project is to build an NLP system for the classification of songs based only on their lyrical content. Ten genres have been selected: 

Commits will be pushed to this repo throughout November and December 2016. Thereafter, this may evolve into an audio-based music recommendation system in tandem.

# Data
The lyrics training and testing datasets are obtained from [the musicmatch website](http://labrosa.ee.columbia.edu/millionsong/musixmatch), genre classification datasets  for the same are obtained from [tagtraum](http://www.tagtraum.com/msd_genre_datasets.html) (CD2 was used here).

The baseline results are as follows (stochastic gradient descent with hinge loss)

| Genre       | Precision       | Recall         |
|-------------|-----------------|----------------|
| Rock        | 0.791749393338  | 0.866803526144 |
| Hip Hop/Rap | 0.254130605822  | 0.868279569892 |
| R&B;        | 0.232746955345  | 0.846456692913 |
| Pop         | 0.416267942584  | 0.710662080825 |
| Electronic  | 0.184131736527  | 0.698070374574 |
| Country     | 0.156942277691  | 0.635101010101 |
| Jazz        | 0.18261608154   | 0.861148197597 |
| Blues       | 0.155754254399  | 0.894039735099 |
| Christian   | 0.0839205637412 | 0.666666666667 |
| Folk        | 0.0553945249597 | 0.747826086957 |