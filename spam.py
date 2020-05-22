# -*- coding: utf-8 -*-
"""
Created on Fri May 22 20:51:27 2020

@author: Ankush
"""

#Step 0 : Import Libraries
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

#Step 1: Import dataset
spam_df = pd.read_csv('emails.csv')
spam_df.head(10)
spam_df.tail(10)
spam_df.describe()
spam_df.info()

#Step 2: Visualize the Dataset
ham = spam_df[spam_df['spam']==0]
ham
spam = spam_df[spam_df['spam']==1]
spam
print('Spam Percentage: =', (len(spam)/len(spam_df))*100, '%')
print('Ham Percentage: =', (len(ham)/len(spam_df))*100, '%')

sns.countplot(spam_df['spam'], label = 'Spam vs Ham')

#Step 3: Create training and testing dataset / Data cleaning
from sklearn.feature_extraction.text import CountVectorizer
vectorizer = CountVectorizer()
spamham_vectorizer = vectorizer.fit_transform(spam_df['text'])

print(vectorizer.get_feature_names())
print(spamham_vectorizer.toarray())
spamham_vectorizer.shape

# Sample Model Training
label = spam_df['spam'].values

from sklearn.naive_bayes import MultinomialNB
classifier = MultinomialNB()
classifier.fit(spamham_vectorizer,label)

testing_sample = ['Free money!!!','Hi Kim, Please let me know if you need any further information. Thanks']
testing_sample_countvectorizer = vectorizer.transform(testing_sample)

test_predict = classifier.predict(testing_sample_countvectorizer)
test_predict

#Step 4: Dividing the data into training and testing dataset prior to model training
X = spamham_vectorizer
y = label

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2)

from sklearn.naive_bayes import MultinomialNB
NB_classifier = MultinomialNB()
NB_classifier.fit(X_train, y_train)

#Step5: Evaluation
from sklearn.metrics import classification_report, confusion_matrix
y_predict_train =  NB_classifier.predict(X_train)
cm = confusion_matrix(y_train, y_predict_train)
sns.heatmap(cm, annot = True)

y_predict_test =  NB_classifier.predict(X_test)
cm1 = confusion_matrix(y_test, y_predict_test)
sns.heatmap(cm1, annot = True)

print(classification_report(y_test, y_predict_test))








