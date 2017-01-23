# Lyrics-based classification of songs 
## Stanford CS 221 Fall 2016 Project
Author: Dhruv Joshi (MS, MS&E 2018)
TA/mentor: Kevin Wu

The purpose of this course project is to build an NLP system for the classification of songs based only on their lyrical content. Ten genres have been selected: 
`genre_labels = ['Rock', 'Pop', 'Hip Hop/Rap', 'R&B', 'Country', 'Jazz', 'Blues', 'Christian']`

Commits will be pushed to this repo throughout November and December 2016.

## Demo
A live demo of the project (using Random Forest) is available [here](http://derbedhruv.webfactional.com/cs221_project). Also, a 3D plot of the GloVe word embeddings is shown here - trained on a 10,000-song corpus and reduced to 3 dimensions. The 3D plotting was done using three.js and was based on amazing work by [Phil Pedruco](http://bl.ocks.org/phil-pedruco/9913243#index.html).

## Data
The lyrics were scraped from [songlyrics.com](http://songlyrics.com), artists A-C with artists having more than 25 songs each, and for which Genre was not NULL. 81940 songs were scraped in total. After cleaning and removing unusable songs, there are 58061 song lyrics in the corpus. "Unusable" songs were the following:
* Songs which had `We do not have the lyrics for ... yet` as their lyrics - these were removed using regex in python
* Songs which had fewer than 100 characters in the lyrics - these were often snippets of lyrics or incorrect lyrics altogether
* Songs which, after running a regex producedure to remove non-english alphanumeric characters, had fewer than 100 characters. This included strange symbols and non-english songs
* Non-english songs were removed by running SQL statements to remove songs with common spanish words in them (From a brief analysis of the common words in each genre, it was found that spanish words were the most common). The artists of all these songs were grouped together and all songs by these artists were removed.

## Assumptions about the data
The following was assumed for the purposes of the project:
* The genre labels from songlyrics.com were assumed to be the ground truth - which would imply that artists, rather than songs, were mapped to genres. This is naturally an incorrect assumption but for the purposes of this project, it was assumed since song lyrics could be easily scraped from songlyrics.com. To increase the validity of this assumption, only those artists were selected that were highly representative of their genre, which would mean that most of their songs would fall in the same genre and style. This information was taken from google and various blogs. Thus, domain knowledge was used to select the training corpus.
* There is only one genre per song - again, this is not entirely correct, but multi-genre classification was not taken into account to keep the model simple. In the online demo, the model's probability estimates are given for each genre/class. The final predicted genre is the one with highest probability.

### Model 1: Random Forest
The results of a random forest trained on 46448 songs and tested on 11613 songs is as follows:

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

The confusion matrix is as follows (Rows are the ground truth and Columns are the predicted values):

|             | Rock | Pop | Hip Hop/Rap | R&B | Country | Jazz | Blues | Christian | 
|-------------|------|-----|-------------|-----|---------|------|-------|-----------| 
| Rock        | 159  | 12  | 4           | 34  | 48      | 23   | 125   | 13        | 
| Pop         | 47   | 64  | 8           | 82  | 105     | 45   | 60    | 29        | 
| Hip Hop/Rap | 16   | 1   | 430         | 16  | 0       | 1    | 6     | 4         | 
| R&B         | 28   | 13  | 18          | 135 | 15      | 17   | 100   | 19        | 
| Country     | 52   | 16  | 2           | 38  | 112     | 42   | 89    | 23        | 
| Jazz        | 17   | 11  | 1           | 20  | 38      | 107  | 56    | 7         | 
| Blues       | 28   | 4   | 0           | 15  | 18      | 9    | 182   | 3         | 
| Christian   | 14   | 13  | 1           | 30  | 45      | 12   | 14    | 205       | 

As we can see, the 'Pop' genre is being misclassified most frequently into Rock, R&B, Country, Jazz and Blues.
