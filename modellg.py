import nltk
from nltk.corpus import twitter_samples

import numpy as np
import pandas as pd
import re  # Library for regular opeartions expressions
import string

from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.tokenize import TweetTokenizer
all_positive_tweets = twitter_samples.strings('positive_tweets.json')
all_negative_tweets = twitter_samples.strings('negative_tweets.json')


def process_tweet(tweet):
    stemmer = PorterStemmer()
    stopwords_english = stopwords.words('english')

    # remove the stock market tickers like $GE
    tweet = re.sub(r'\$\w*', '', tweet)

    # remove the old style retweet text RT
    tweet = re.sub(r'^RT[\s]+', '', tweet)

    # remove the hyperlinks
    tweet = re.sub(r'https?:\/\/.*[\r\n]*', '', tweet)

    # removing # symbol
    tweet = re.sub(r'#', '', tweet)

    tokenizer = TweetTokenizer(preserve_case=False, reduce_len=True, strip_handles=True)
    tweet_tokens = tokenizer.tokenize(tweet)

    tweets_clean = []

    for word in tweet_tokens:
        if (word not in stopwords_english and
                word not in string.punctuation):
            stem_word = stemmer.stem(word)  # stemming
            tweets_clean.append(stem_word)

    return tweets_clean
def build_freqs(tweets, ys):
  yslist = np.squeeze(ys).tolist()

  freqs = {}
  for y, tweet in zip(yslist, tweets):
    for word in process_tweet(tweet):
      pair = (word, y)
      if pair in freqs:
        freqs[pair]+=1
      else:
        freqs[pair] =1
  return freqs
train_pos = all_positive_tweets[:4000]
test_pos = all_positive_tweets[4000:]

train_neg = all_negative_tweets[:4000]
test_neg = all_negative_tweets[4000:]

train_x = train_pos + train_neg
test_x = test_pos + test_neg

# making a numpy labels array

train_y = np.append(np.ones((len(train_pos), 1)), np.zeros((len(train_neg), 1)), axis=0)
test_y = np.append(np.ones((len(test_pos), 1)), np.zeros((len(test_neg), 1)), axis=0)

freqs = build_freqs(train_x, train_y)

print('Type of freqs : ', type(freqs))
print('Length of freqs : ', len(freqs))

def sigmoid(z):
  h = 1/(1+np.exp(-z))
  return h

def gradientDescent(x, y, theta, alpha, num_iters):
  m = x.shape[0]
  for i in range(num_iters):
    z = np.dot(x, theta)
    h = sigmoid(z)

    # cost function
    J = (-1/m) * (np.dot(y.T, np.log(h)) + np.dot((1-y).T, np.log(1-h)))

    # update the weights theta
    theta = theta - (alpha/m)* (np.dot(x.T, h-y))

  J = float(J)
  return J, theta

def extract_features(tweet, freqs):

  # Process the tweet
  word_l = process_tweet(tweet)

  x = np.zeros((1, 3))

  # bias term is set to 1
  x[0, 0] = 1

  for word in word_l:
    x[0, 1] += freqs.get((word, 1), 0)
    x[0, 2] += freqs.get((word, 0), 0)

  assert(x.shape==(1, 3))
  return x
# collect the features 'x' and stack them into a matrix 'X'

X = np.zeros((len(train_x), 3))
for i in range(len(train_x)):
  X[i, :] = extract_features(train_x[i], freqs)

# training levels corrosponding to X
Y = train_y

# Applying Gradient Descent
J, theta = gradientDescent(X, Y, np.zeros((3, 1)), 1e-9, 1500)

print(f"The cost after training is {J:.8f}.")
print(f"The resulting vector of weights is {[round(t, 8) for t in np.squeeze(theta)]}")

def predict_tweet(tweet, freqs, theta):

  x = extract_features(tweet, freqs)

  # make prediction
  y_pred = sigmoid(np.dot(x, theta))

  return y_pred


def test_logistic_regression(test_x, test_y, feqs, theta):
    # empty list for storing predictions
    y_hat = []

    for tweet in test_x:
        y_pred = predict_tweet(tweet, freqs, theta)

        if y_pred > 0.5:
            y_hat.append(1)
        else:
            y_hat.append(0)

    accuracy = (y_hat == np.squeeze(test_y)).sum() / len(test_x)
    return accuracy


accuracy = test_logistic_regression(test_x, test_y, freqs, theta)
print('The accuracy of Logistic Regression is :', accuracy)