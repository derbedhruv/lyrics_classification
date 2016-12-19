# Lyrics-based classification of songs 
## Stanford CS 221 Fall 2016 Mini Project
Author: Dhruv Joshi

The purpose of this course mini-project is to build an NLP system for the classification of songs based only on their lyrical content. Ten genres have been selected: 
`genre_labels = ['Rock', 'Pop', 'Hip Hop/Rap', 'R&B', 'Country', 'Jazz', 'Blues', 'Christian']`

Commits will be pushed to this repo throughout November and December 2016. Thereafter, this may evolve into an audio-based music recommendation system in tandem.

## Data
The lyrics were scraped from [songlyrics.com](http://songlyrics.com), artists A-C with artists having more than 25 songs each, and for which Genre was not NULL. 81940 songs were scraped in total. After cleaning and removing unusable songs, there are 58061 song lyrics in the corpus.

The latest results are as follows (Random Forest Classifier, trained on 46448 songs and tested on 11613 songs)

| Genre       	| Precision       	| Recall         	|
|-------------	|-----------------	|----------------	|
| Rock        	| 0.47  			| 0.72 				|
| Pop			| 0.57  			| 0.39 				|
| Hip Hop/Rap   | 0.76  			| 0.76 				|
| R&B         	| 0.53 			 	| 0.48 				|
| Country     	| 0.45  			| 0.57 				|
| Jazz        	| 0.60   			| 0.38 				|
| Blues       	| 0.75  			| 0.43 				|
| Christian   	| 0.64 				| 0.55 				|


## Demo
A live demo of the project (using Random Forest) is available [here](http://derbedhruv.webfactional.com/cs221_project).
