import re

import nltk
import numpy as np
import pandas as pd
import tensorflow as tf
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize
from sklearn.metrics import (accuracy_score, classification_report,
                             confusion_matrix)
from sklearn.model_selection import train_test_split
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.layers import LSTM, Dense, Embedding
from tensorflow.keras.models import Sequential
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import one_hot

trueNewsDf = pd.read_csv("data/true_fake_news/True.csv")
fakeNewsDf = pd.read_csv("data/true_fake_news/Fake.csv")


trueNewsDf["label"] = 1
fakeNewsDf["label"] = 0

df = pd.concat([trueNewsDf, fakeNewsDf], ignore_index=True)
df =df.dropna()

# Data splits
trainDf, testDf = train_test_split(df, test_size=0.2, random_state=42)

trainData = trainDf["title"]
trainLabels = trainDf["label"]
testData = testDf["title"]
testLabels = testDf["label"]

#Preprocessing
stemmer =PorterStemmer()
stopWords = set(stopwords.words("english"))
trainDataset = []
for text in trainData:
    cleanedWords = re.sub("[^a-zA-Z]", " ", text)
    words = word_tokenize(cleanedWords)
    stemmedWords = [stemmer.stem(word).lower() for word in words if word not in stopWords]
    trainDataset.append(" ".join(stemmedWords))

testDataset = []
for text in testData:
    cleanedWords = re.sub("[^a-zA-Z]", " ", text)
    words = word_tokenize(cleanedWords)
    stemmedWords = [stemmer.stem(word).lower() for word in words if word not in stopWords]
    testDataset.append(" ".join(stemmedWords))

#onehot representation
vocabSize = 5000
oneHotTrain = [one_hot(word, vocabSize) for word in trainDataset]
oneHotTest = [one_hot(word, vocabSize) for word in testDataset]

#padding
maxSentenceLength = max([len(sentence.split(" ")) for sentence in trainDataset])
embeddedWordsTrain = pad_sequences(oneHotTrain, padding="pre", maxlen=maxSentenceLength)
embeddedWordsTest = pad_sequences(oneHotTest, padding="pre", maxlen=maxSentenceLength)

#model
vectorFeaturesSize = 20
model = Sequential()
model.add(Embedding(vocabSize, vectorFeaturesSize, input_length=maxSentenceLength))
model.add(LSTM(64, dropout=0.3, recurrent_dropout=0.3))
model.add(Dense(1, activation="sigmoid"))
model.compile(loss="binary_crossentropy", optimizer="adam", metrics=["accuracy"])
model.summary()

earlyStop = EarlyStopping(monitor="val_loss", patience=2, restore_best_weights=True)
model.fit(embeddedWordsTrain, trainLabels, validation_data=(embeddedWordsTest, testLabels), epochs=10, batch_size=64, callbacks=[earlyStop])
yProb = model.predict(embeddedWordsTest)
pred = (yProb > 0.5).astype(int)
accuracyScore = accuracy_score(testLabels, pred)
confusionMatrix = confusion_matrix(testLabels, pred)
classificationReport = classification_report(testLabels, pred)
print("accuracy: ", accuracyScore)
print("confusion_matrix: ", confusionMatrix)
print("classification_report: ", classificationReport)