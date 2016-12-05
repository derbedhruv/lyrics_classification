# Lyrics-based classification of songs 
## Stanford CS 221 Fall 2016 Mini Project
Author: Dhruv Joshi

The purpose of this course mini-project is to build an NLP system for the classification of songs based only on their lyrical content. Ten genres have been selected: 
`genre_labels = ['Rock', 'Pop', 'Hip Hop/Rap', 'R&B;', 'Electronic', 'Country', 'Jazz', 'Blues', 'Christian', 'Folk']`

Commits will be pushed to this repo throughout November and December 2016. Thereafter, this may evolve into an audio-based music recommendation system in tandem.

# Data
The lyrics were scraped from [songlyrics.com](http://songlyrics.com), artists A-C with artists having more than 25 songs each, and for which Genre was not NULL. 81940 songs were scraped in total.

The baseline results are as follows (stochastic gradient descent with hinge loss, trained on 65552 songs and tested on 16388 songs)

| Genre       	| Precision       	| Recall         	|
|-------------	|-----------------	|----------------	|
| Rock        	| 0.47  			| 0.72 				|
| Hip Hop/Rap 	| 0.57  			| 0.39 				|
| R&B       	| 0.76  			| 0.76 				|
| Pop         	| 0.53 			 	| 0.48 				|
| Country     	| 0.45  			| 0.57 				|
| Jazz        	| 0.60   			| 0.38 				|
| Blues       	| 0.75  			| 0.43 				|
| Christian   	| 0.64 				| 0.55 				|