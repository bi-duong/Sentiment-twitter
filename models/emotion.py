# import pandas as pd
# import re
# import numpy as np
# import seaborn as sns
# import neattext.functions as nfx
# from sklearn.pipeline import Pipeline
# import pickle
#
# # loading ML packages
#
# from sklearn.linear_model import LogisticRegression
#
# # transformers
# from sklearn.feature_extraction.text import CountVectorizer
# from sklearn.model_selection import train_test_split
# from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
# from sklearn.feature_extraction.text import TfidfVectorizer
#
# # load dataset
# data = pd.read_csv("./data/emotion.csv")
#
# data['Emotion'].value_counts()
#
# # data cleaning
# dir(nfx)
#
# # user handles
#
# data['clean_text'] = data['Text'].apply(nfx.remove_userhandles)
#
# data['clean_text'] = data['clean_text'].apply(nfx.remove_stopwords)
#
#
#
# vectr = TfidfVectorizer(ngram_range=(1, 2), min_df=1, )
# vectr.fit(data['clean_text'])
#
# # features and labels
# xfeatures = data['clean_text']
# ylabels = data['Emotion']
#
# # split data
# x_train, x_test, y_train, y_test = train_test_split(xfeatures, ylabels, test_size=0.3, random_state=42)
#
# #Logistic Regression Pipeline
# pipe_lr = Pipeline(steps=[('cv', CountVectorizer(lowercase=False)),('lr',LogisticRegression())])
#
# #train and fit data
# pipe_lr.fit(x_train,y_train)
#
#
#
# # check accuracy
# score = pipe_lr.score(x_test, y_test)
#
# # make prediction
#
# # make prediction
# ex1 = "I am not feeling well"
#
# pred = pipe_lr.predict([ex1])
# print(pred)
#
# # to know classes and prediction probability
# classes = pipe_lr.classes_
#
# prob = pipe_lr.predict_proba([ex1])
# print(prob)
#
# # Saving model to disk
# pickle.dump(pipe_lr, open('./models/emotion_pred_model.pkl','wb'))
#
# # Loading model to compare the results (read)
# loaded_model = pickle.load(open('./models/emotion_pred_model.pkl','rb'))
# loaded_model.predict([ex1])
