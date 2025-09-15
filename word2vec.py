import re

import gensim.downloader as api
import nltk
import numpy as np
import pandas as pd
from gensim.models import Word2Vec
from gensim.utils import simple_preprocess
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import sent_tokenize, word_tokenize
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (accuracy_score, classification_report,
                             confusion_matrix)
from sklearn.model_selection import train_test_split
from tqdm import tqdm

wordVector = api.load("word2vec-google-news-300")

messages = pd.read_csv("data/sms_spam_collection/SMSSpamCollection", sep="\t", names=["label", "message"])

words = []
for i in range(len(messages)):
    sentences = sent_tokenize(messages["message"][i])
    for sentence in sentences:
        words.append(simple_preprocess(sentence))

model =wordVector
vocabs = model.key_to_index


def avgWord2Vec(doc):
    validWords = [model[word] for word in doc if word in vocabs]
    if not validWords:
        return np.zeros(model.vector_size)
    return np.mean(validWords, axis=0)

X = []
for i in tqdm(range(len(messages))):
    sentences = sent_tokenize(messages["message"][i])
    sentenceVecs = [avgWord2Vec(simple_preprocess(s)) for s in sentences]
    X.append(np.mean(sentenceVecs, axis=0))

data = np.array(X)
print(data.shape)
labels = pd.get_dummies(messages["label"])
labels = labels.iloc[:,1].values

df = pd.DataFrame(data)
trainX, testX, trainY, testY = train_test_split(df, labels, test_size=0.2)


classifier = RandomForestClassifier()
classifier.fit(trainX, trainY)
pred = classifier.predict(testX)


accuracy = accuracy_score(testY, pred)
confusionMatrix = confusion_matrix(testY, pred)
classificationReport = classification_report(testY, pred)
print("accuracy: ", accuracy)
print("confusion matrix: ", confusionMatrix)
print("classification report: ", classificationReport)