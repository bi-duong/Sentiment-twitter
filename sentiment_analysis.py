import re
import tweepy
import pandas as pd
from textblob import TextBlob
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
# from wordcloud import WordCloud
import numpy as np
import matplotlib

from modellg import predict_tweet, freqs, theta

matplotlib.use('Agg')
import matplotlib.pyplot as plt
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from flask import Flask, render_template, redirect, url_for, request
from modelnb import naive_bayes_predict, logprior, loglikelihood
import nltk
#nltk.download('vader_lexicon')
# from transformers import pipeline
vader = SentimentIntensityAnalyzer()
emotion_pred_model = pickle.load(open('./models/emotion_pred_model.pkl','rb'))
with open('models/model2_pkl' , 'rb') as f:
    model_transformers = pickle.load(f)
# lay cac API
API_Key = "YOfX4r7g3YCy5D2vE3BUG115I"
API_Key_Secret = "GbTTmWA5JlEyPMJF9WCTMcyPHeNcXjHBREyjSYiX4iIyHwaLlZ"
Access_Token = "1502904942344179718-or4wUdWtudoY9HX3hjLjJSmQpvCyBx"
Access_Token_Secret = "AUFrsoATq5EBBNbpB4o3YdaIxppP1ghq3mfqtMZjGp9PC"
authenticate = tweepy.OAuthHandler(API_Key, API_Key_Secret)
authenticate.set_access_token(Access_Token, Access_Token_Secret)
api = tweepy.API(authenticate, wait_on_rate_limit=True)

#
app = Flask(__name__ , template_folder='templates')
app.static_folder = 'static'
#lam sach du lieu
def cleanText(text):
    text = re.sub(r'@[A-Za-z0-9]+', '',text)   # For Removing mentions
    text = re.sub(r'#','',text) # For Removing hashtags
    text = re.sub(r'RT[\s]+','',text) #Remove Re-Tweets
    text = re.sub(r'https?:\/\/\S+','',text) # Remove Hyperlinks
    return text

#tinh do uu tien bang textblob
def getSubjectivity(text):
    return round(TextBlob(text).sentiment.subjectivity,3)

def getPolarity(text):
    return round(TextBlob(text).sentiment.polarity,3)
#tinh do uu tien bang vader
def getPolarity_vader(text):
    return round(vader.polarity_scores(text)['compound'],3)
# tinh gia tri cua cau
def analysis(score):
    if (score > 0.5):
        return 'Positive'
    elif (score < 0.5):
        return 'Negative'
    else:
        return 'Neutral'
def analysisText(score):
    if (score > 0.9):
        return 'Positive 😄'
    elif (score > 0.8) & (score<=0.9):
        return 'Negative fd😄'
    elif (score > 0.7)&(score <= 0.8):
        return 'Negadfgftive 😄'
    else:
        return 'Neutral 😄'
##################user##################################################################
def get_user_tweets(api,user_name,count=5):
    count = int(count)

    posts = tweepy.Cursor(api.user_timeline, screen_name=user_name,lang ='en', count=5,       tweet_mode="extended" ).items(count)
    df = pd.DataFrame([tweet.full_text for tweet in posts], columns=['Tweets'])

    df['Tweets'] = df['Tweets'].apply(cleanText)
    df['Subjectivity'] = df['Tweets'].apply(getSubjectivity)
    df['Polarity'] = df['Tweets'].apply(getPolarity)
    df['Sentiment'] = df['Polarity'].apply(analysis)
    return df
def get_user_tweets_vader(api,user_name,count=5):
    count = int(count)
    posts = tweepy.Cursor(api.user_timeline, screen_name=user_name,lang ='en', count=5,       tweet_mode="extended" ).items(count)
    df = pd.DataFrame([tweet.full_text for tweet in posts], columns=['Tweets'])
    df['Tweets'] = df['Tweets'].apply(cleanText)
    df['Polarity'] = df['Tweets'].apply(getPolarity_vader)
    df['Sentiment'] = df['Polarity'].apply(analysis)
    return df
def get_user_tweets_nb(api,user_name,count=5):
    count = int(count)
    posts = tweepy.Cursor(api.user_timeline, screen_name=user_name,lang ='en', count=5,       tweet_mode="extended" ).items(count)
    df = pd.DataFrame([tweet.full_text for tweet in posts], columns=['Tweets'])
    df['Tweets'] = df['Tweets'].apply(cleanText)
    addpol = []
    for o in df['Tweets']:
        g = naive_bayes_predict(o, logprior, loglikelihood)
        addpol.append(round(g,3))
    df['Polarity'] = addpol
    df['Sentiment'] = df['Polarity'].apply(analysis)
    return df
def get_user_tweets_linegres(api,user_name,count=5):
    count = int(count)
    posts = tweepy.Cursor(api.user_timeline, screen_name=user_name,lang ='en', count=5,       tweet_mode="extended" ).items(count)
    df = pd.DataFrame([tweet.full_text for tweet in posts], columns=['Tweets'])
    df['Tweets'] = df['Tweets'].apply(cleanText)
    addpol = []
    for o in df['Tweets']:
        g = predict_tweet(o, freqs, theta)
        addpol.append(g)
    df['Polarity'] = addpol
    df['Sentiment'] = df['Polarity'].astype(float).apply(analysis)
    return df
def get_user_tweets_transformer(api,user_name,count=5):
    count = int(count)
    posts = tweepy.Cursor(api.user_timeline, screen_name=user_name,lang ='en', count=5,       tweet_mode="extended" ).items(count)
    df = pd.DataFrame([tweet.full_text for tweet in posts], columns=['Tweets'])
    df['Tweets'] = df['Tweets'].apply(cleanText)
    addpol = []
    for o in df['Tweets']:
        g = model_transformers.predict(o)
        addpol.append(g)
    df['Polarity'] = addpol
    df['Sentiment'] = df['Polarity'].str[0].str['label']
    return df
#####################Hastag#####################################################
def get_hashtag_tweets(api,hashtag,count=5):
    count = int(count)
    posts = tweepy.Cursor(api.search_tweets, q=hashtag, count=100,lang ='en', tweet_mode="extended").items(count)
    df = pd.DataFrame([tweet.full_text for tweet in posts], columns=['Tweets'])

    df['Tweets'] = df['Tweets'].apply(cleanText)
    df['Subjectivity'] = df['Tweets'].apply(getSubjectivity)
    df['Polarity'] = df['Tweets'].apply(getPolarity)
    df['Sentiment'] = df['Polarity'].apply(analysis)
    return df
def get_hashtag_tweets_vader(api,hashtag,count=5):
    count = int(count)
    posts = tweepy.Cursor(api.search_tweets, q=hashtag, count=100,lang ='en', tweet_mode="extended").items(count)
    df = pd.DataFrame([tweet.full_text for tweet in posts], columns=['Tweets'])
    df['Tweets'] = df['Tweets'].apply(cleanText)
    df['Polarity'] = df['Tweets'].apply(getPolarity_vader)
    df['Sentiment'] = df['Polarity'].apply(analysis)
    return df
def get_hashtag_tweets_nb(api,hashtag,count=5):
    count = int(count)
    posts = tweepy.Cursor(api.search_tweets, q=hashtag, count=100,lang ='en', tweet_mode="extended").items(count)
    df = pd.DataFrame([tweet.full_text for tweet in posts], columns=['Tweets'])
    df['Tweets'] = df['Tweets'].apply(cleanText)
    addpol = []
    for o in df['Tweets']:
        g = naive_bayes_predict(o, logprior, loglikelihood)
        addpol.append(round(g,3))
    df['Polarity'] = addpol
    df['Sentiment'] = df['Polarity'].apply(analysis)
    return df
def get_hashtag_tweets_linegres(api,hashtag,count=5):
    count = int(count)
    posts = tweepy.Cursor(api.search_tweets, q=hashtag, count=100,lang ='en', tweet_mode="extended").items(count)
    df = pd.DataFrame([tweet.full_text for tweet in posts], columns=['Tweets'])
    df['Tweets'] = df['Tweets'].apply(cleanText)
    addpol = []
    for o in df['Tweets']:
        g = predict_tweet(o, freqs, theta)
        addpol.append(g)
    df['Polarity'] = addpol
    df['Sentiment'] = df['Polarity'].astype(float).apply(analysis)
    return df
def get_hashtag_tweets_transformer(api,hashtag,count=5):
    count = int(count)
    posts = tweepy.Cursor(api.search_tweets, q=hashtag, count=100,lang ='en', tweet_mode="extended").items(count)
    df = pd.DataFrame([tweet.full_text for tweet in posts], columns=['Tweets'])
    df['Tweets'] = df['Tweets'].apply(cleanText)
    addpol = []
    for o in df['Tweets']:
        g = model_transformers.predict(o)
        addpol.append(g)
    df['Polarity'] = addpol
    df['Sentiment'] = df['Polarity'].str[0].str['label']
    return df

##=======================Plot#######################################
def plot(df,name):
    plt.title('Sentiment Analysis Result of '+name)
    plt.ylabel('Counts')
    colors = ['#ff9999', '#66b3ff', '#99ff99']
    df['Sentiment'].value_counts().plot(kind='bar')
    plt.xticks(rotation=0)
    plt.savefig('static/my_plot.png')
    plt.switch_backend('agg')
def line_chart(df,name):
    plt.title('Sentiment Analysis Result of ' + name)
    Neutral = len(df[df['Sentiment'] == 'Neutral'])
    Negative = len(df[df['Sentiment'] == 'Negative'])
    Positive = len(df[df['Sentiment'] == 'Positive'])
    values = [Negative, Positive, Neutral]
    plt.plot(values, linestyle='dotted')
    plt.savefig('static/my_line.png')
    plt.switch_backend('agg')
def multi_line_chart(df,name):
    plt.title('Sentiment Analysis Result of ' + name)
    # plt.ylabel('Counts')
    # colors = ['#ff9999', '#66b3ff', '#99ff99']
    df['Sentiment'].value_counts().plot(kind='barh')
    # plt.xticks(rotation=0)

    # plt.plot(df['Sentiment'] == 'Negative', color='purple', linestyle='dotted')
    # plt.plot(df['Sentiment'] == 'Positive',color='steelblue', linestyle='dashed')
    # plt.plot(df['Sentiment'] == 'Neutral')

    plt.savefig('static/multi_line.png')
    plt.switch_backend('agg')
def pie(df,name):
    plt.title('Sentiment Analysis Result of '+name)
    Neutral = len(df[df['Sentiment'] == 'Neutral'])
    Negative = len(df[df['Sentiment'] == 'Negative'])
    Positive = len(df[df['Sentiment'] == 'Positive'])
    labels = ['Negative', 'Positive', 'Neutral']
    values = [Negative, Positive, Neutral]
    colors = ['#ff9999', '#66b3ff', '#99ff99']
    explode = (0.05, 0.05, 0.05)
    plt.pie(values,labels=labels, startangle=90, autopct='%1.1f%%',shadow=True,colors = colors,explode = explode)
    plt.savefig('static/my_pie.png')
    plt.switch_backend('agg')
#==============================================

#goi cac route
@app.route('/')
def home():
  return render_template("index.html")
@app.route('/layout')
def layout():
  return render_template("layout.html")
@app.route('/textblob')
def textblob():
  return render_template("textblob.html")
@app.route('/text')
def text():
  return render_template("text.html")
@app.route('/textvader')
def textvader():
  return render_template("textvader.html")
@app.route('/table')
def table():
  return render_template("table.html")
@app.route('/nb')
def nb():
  return render_template("nb.html")
@app.route('/lg')
def lg():
  return render_template("lg.html")
@app.route('/transformers')
def transformers():
  return render_template("transformers.html")
### xu ly du lieu
##user
@app.route("/predict_user", methods=['POST','GET'])
def predict_user():
    if request.method == 'POST':
        user = request.form['user_name']
        count = request.form['count']
        fetched_tweets = get_user_tweets(api, user, count)
        plot(fetched_tweets, user)
        pie(fetched_tweets, user)
        multi_line_chart(fetched_tweets, user)
        line_chart(fetched_tweets, user)

        positive_tweets = fetched_tweets[fetched_tweets.Sentiment == 'Positive']
        negative_tweets = fetched_tweets[fetched_tweets.Sentiment == 'Negative']
        x = round((positive_tweets.shape[0] / fetched_tweets.shape[0]) * 100, 1)
        y = round((negative_tweets.shape[0] / fetched_tweets.shape[0]) * 100, 1)
        z = round(100 - (x + y), 1)

        fetched_tweets = fetched_tweets.to_dict('records')

        return render_template('result_user.html', result=fetched_tweets, pos=x,neg=y,neu=z)
@app.route("/predict_user_vader", methods=['POST','GET'])
def predict_user_vader():
    if request.method == 'POST':
        user = request.form['user_name']
        count = request.form['count']
        fetched_tweets = get_user_tweets_vader(api, user, count)
        plot(fetched_tweets, user)
        pie(fetched_tweets, user)
        multi_line_chart(fetched_tweets, user)
        line_chart(fetched_tweets, user)

        positive_tweets = fetched_tweets[fetched_tweets.Sentiment == 'Positive']
        negative_tweets = fetched_tweets[fetched_tweets.Sentiment == 'Negative']
        x = round((positive_tweets.shape[0] / fetched_tweets.shape[0]) * 100, 1)
        y = round((negative_tweets.shape[0] / fetched_tweets.shape[0]) * 100, 1)
        z = round(100 - (x + y), 1)

        fetched_tweets = fetched_tweets.to_dict('records')

        return render_template('result_user.html', result=fetched_tweets, pos=x,neg=y,neu=z)
@app.route("/predict_user_nb", methods=['POST','GET'])
def predict_user_nb():
    if request.method == 'POST':
        user = request.form['user_name']
        count = request.form['count']
        fetched_tweets = get_user_tweets_nb(api, user, count)
        plot(fetched_tweets, user)
        pie(fetched_tweets, user)
        multi_line_chart(fetched_tweets, user)
        line_chart(fetched_tweets, user)

        positive_tweets = fetched_tweets[fetched_tweets.Sentiment == 'Positive']
        negative_tweets = fetched_tweets[fetched_tweets.Sentiment == 'Negative']
        x = round((positive_tweets.shape[0] / fetched_tweets.shape[0]) * 100, 1)
        y = round((negative_tweets.shape[0] / fetched_tweets.shape[0]) * 100, 1)
        z = round(100 - (x + y), 1)

        fetched_tweets = fetched_tweets.to_dict('records')

        return render_template('result_user.html', result=fetched_tweets, pos=x,neg=y,neu=z)
@app.route("/predict_user_lg", methods=['POST','GET'])
def predict_user_lg():
    if request.method == 'POST':
        user = request.form['user_name']
        count = request.form['count']
        fetched_tweets = get_user_tweets_linegres(api, user, count)
        plot(fetched_tweets, user)
        pie(fetched_tweets, user)
        multi_line_chart(fetched_tweets, user)
        line_chart(fetched_tweets, user)

        positive_tweets = fetched_tweets[fetched_tweets.Sentiment == 'Positive']
        negative_tweets = fetched_tweets[fetched_tweets.Sentiment == 'Negative']
        x = round((positive_tweets.shape[0] / fetched_tweets.shape[0]) * 100, 1)
        y = round((negative_tweets.shape[0] / fetched_tweets.shape[0]) * 100, 1)
        z = round(100 - (x + y), 1)

        fetched_tweets = fetched_tweets.to_dict('records')

        return render_template('result_user.html', result=fetched_tweets, pos=x,neg=y,neu=z)
@app.route("/predict_user_transformer", methods=['POST','GET'])
def predict_user_transformer():
    if request.method == 'POST':
        user = request.form['user_name']
        count = request.form['count']
        fetched_tweets = get_user_tweets_transformer(api, user, count)
        plot(fetched_tweets,user)

        positive_tweets = fetched_tweets[fetched_tweets.Sentiment == 'POSITIVE']
        negative_tweets = fetched_tweets[fetched_tweets.Sentiment == 'NEGATIVE']
        x = round((positive_tweets.shape[0] / fetched_tweets.shape[0]) * 100, 1)
        y = round((negative_tweets.shape[0] / fetched_tweets.shape[0]) * 100, 1)
        z = round(100 - (x + y), 1)

        fetched_tweets = fetched_tweets.to_dict('records')

        return render_template('result_user.html', result=fetched_tweets, pos=x,neg=y,neu=z)
### hashtag
@app.route("/predict_tag", methods=['POST','GET'])
def predict_tag():
    if request.method == 'POST':
        hash = request.form['hashtag']
        count = request.form['count']
        fetched_tweets = get_hashtag_tweets(api,hash,count)
        plot(fetched_tweets, hash)
        pie(fetched_tweets, hash)
        multi_line_chart(fetched_tweets, hash)
        line_chart(fetched_tweets, hash)
        positive_tweets = fetched_tweets[fetched_tweets.Sentiment == 'Positive']
        negative_tweets = fetched_tweets[fetched_tweets.Sentiment == 'Negative']
        x = round((positive_tweets.shape[0] / fetched_tweets.shape[0]) * 100, 1)
        y = round((negative_tweets.shape[0] / fetched_tweets.shape[0]) * 100, 1)
        z = round(100 - (x + y), 1)

        fetched_tweets = fetched_tweets.to_dict('records')

        return render_template('result_user.html', result=fetched_tweets, pos=x, neg=y, neu=z)

@app.route("/predict_tag_vader", methods=['POST','GET'])
def predict_tag_vader():
    if request.method == 'POST':
        hash = request.form['hashtag vader']
        count = request.form['count']
        fetched_tweets = get_hashtag_tweets_vader(api, hash, count)
        plot(fetched_tweets, hash)
        pie(fetched_tweets, hash)
        multi_line_chart(fetched_tweets, hash)
        line_chart(fetched_tweets, hash)
        positive_tweets = fetched_tweets[fetched_tweets.Sentiment == 'Positive']
        negative_tweets = fetched_tweets[fetched_tweets.Sentiment == 'Negative']
        x = round((positive_tweets.shape[0] / fetched_tweets.shape[0]) * 100, 1)
        y = round((negative_tweets.shape[0] / fetched_tweets.shape[0]) * 100, 1)
        z = round(100 - (x + y), 1)
        fetched_tweets = fetched_tweets.to_dict('records')
        return render_template('result_user.html', result=fetched_tweets, pos=x, neg=y, neu=z)

@app.route("/predict_tag_nb", methods=['POST','GET'])
def predict_tag_nb():
    if request.method == 'POST':
        hash = request.form['hashtag nb']
        count = request.form['count']
        fetched_tweets = get_hashtag_tweets_nb(api, hash, count)
        plot(fetched_tweets, hash)
        pie(fetched_tweets, hash)
        multi_line_chart(fetched_tweets, hash)
        line_chart(fetched_tweets, hash)
        positive_tweets = fetched_tweets[fetched_tweets.Sentiment == 'Positive']
        negative_tweets = fetched_tweets[fetched_tweets.Sentiment == 'Negative']
        x = round((positive_tweets.shape[0] / fetched_tweets.shape[0]) * 100, 1)
        y = round((negative_tweets.shape[0] / fetched_tweets.shape[0]) * 100, 1)
        z = round(100 - (x + y), 1)
        fetched_tweets = fetched_tweets.to_dict('records')
        return render_template('result_user.html', result=fetched_tweets, pos=x, neg=y, neu=z)


@app.route("/predict_tag_linegres", methods=['POST','GET'])
def predict_tag_linegres():
    if request.method == 'POST':
        hash = request.form['hashtag linegres']
        count = request.form['count']
        fetched_tweets = get_hashtag_tweets_linegres(api, hash, count)
        plot(fetched_tweets, hash)
        pie(fetched_tweets, hash)
        multi_line_chart(fetched_tweets, hash)
        line_chart(fetched_tweets, hash)
        positive_tweets = fetched_tweets[fetched_tweets.Sentiment == 'Positive']
        negative_tweets = fetched_tweets[fetched_tweets.Sentiment == 'Negative']
        x = round((positive_tweets.shape[0] / fetched_tweets.shape[0]) * 100, 1)
        y = round((negative_tweets.shape[0] / fetched_tweets.shape[0]) * 100, 1)
        z = round(100 - (x + y), 1)
        fetched_tweets = fetched_tweets.to_dict('records')
        return render_template('result_user.html', result=fetched_tweets, pos=x, neg=y, neu=z)

@app.route("/predict_tag_transformers", methods=['POST','GET'])
def predict_tag_transformerss():
    if request.method == 'POST':
        hash = request.form['hashtag linegres']
        count = request.form['count']
        fetched_tweets = get_hashtag_tweets_transformer(api, hash, count)
        plot(fetched_tweets, hash)
        # pie(fetched_tweets, hash)
        # multi_line_chart(fetched_tweets, hash)
        # line_chart(fetched_tweets, hash)
        positive_tweets = fetched_tweets[fetched_tweets.Sentiment == 'POSITIVE']
        negative_tweets = fetched_tweets[fetched_tweets.Sentiment == 'NEGATIVE']
        x = round((positive_tweets.shape[0] / fetched_tweets.shape[0]) * 100, 1)
        y = round((negative_tweets.shape[0] / fetched_tweets.shape[0]) * 100, 1)
        z = round(100 - (x + y), 1)
        fetched_tweets = fetched_tweets.to_dict('records')
        return render_template('result_user.html', result=fetched_tweets, pos=x, neg=y, neu=z)

#####text data#############################
@app.route("/predict_sentence",methods=['POST','GET'])
def predict_text():
    if request.method== 'POST':
        sentence = request.form['sentence']
        Subjectivity = getSubjectivity(sentence)
        Polarity = getPolarity(sentence)
        Sentiment = analysisText(Polarity)
        return render_template('result_sentence.html',sentence=sentence,Subjectivity=Subjectivity,Polarity=Polarity,Sentiment=Sentiment)
###text data model h5
@app.route('/emotion', methods=['GET','POST'])
def emotionanalysis():
    output = 'Result'
    if request.method == 'POST':
        keyword = request.form.get('emo_text')
        output = emotion_pred_model.predict([keyword])

        if output == 'joy':
            output = "Joy 😄"
        elif output =='neutral':
            output = "neutral 🙂"
        elif output =='sadness':
            output = "sadness 😔"
        elif output =='fear':
            output = "fear 😱"
        elif output =='surprise':
            output = "surprise 😃"
        elif output =='anger':
            output = "anger 😠"

    return render_template('emotion.html', prediction=f'{output}')
if __name__ == '__main__':

    app.debug = True
    app.run()