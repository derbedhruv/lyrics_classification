from train_lyrics import *

data = get_data()

n = len(data)
train = data.sample(n=round(0.8*n))
test = data.sample(n=round(0.2*n))

# save the train and test data
train.to_csv('train.csv')
test.to_csv('test.csv')

print 'saved!'
