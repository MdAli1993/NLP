import re

import pandas as pd
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn.metrics import (accuracy_score, classification_report,
                             confusion_matrix)
from sklearn.model_selection import train_test_split

trueNewsDf = pd.read_csv("data/true_fake_news/True.csv")
fakeNewsDf = pd.read_csv("data/true_fake_news/Fake.csv")


trueNewsDf["label"] = 1
fakeNewsDf["label"] = 0

df = pd.concat([trueNewsDf, fakeNewsDf], ignore_index=True)

# Data splits
trainDf, testDf = train_test_split(df, test_size=0.2, random_state=42)

trainData = trainDf["text"]
trainLabels = trainDf["label"]
testData = testDf["text"]
testLabels = testDf["label"]

# Train data preprocessing
stemmer = PorterStemmer()
stopWords = set(stopwords.words("english"))
trainDataset = []
for text in trainData:
    cleanedWords = re.sub("[^a-zA-Z]", " ", text)
    words = word_tokenize(cleanedWords)
    stemmedWords = [stemmer.stem(word).lower() for word in words if word not in stopWords]
    trainDataset.append(" ".join(stemmedWords))

# TFIDF
countVectorizeTFIDF = TfidfVectorizer(max_features=5000, ngram_range=(1,3))
X = countVectorizeTFIDF.fit_transform(trainDataset)

# Model Training
passiveAggressiveClassifier = PassiveAggressiveClassifier(max_iter=50)
passiveAggressiveClassifier.fit(X, trainLabels)

# Test dtaa preprocessing
testDataset = []
for text in testData:
    cleanedWords = re.sub("[^a-zA-Z]", " ", text)
    words = word_tokenize(cleanedWords)
    stemmedWords = [stemmer.stem(word).lower() for word in words if word not in stopWords]
    testDataset.append(" ".join(stemmedWords))
Y = countVectorizeTFIDF.transform(testDataset)
# Predictions
pred = passiveAggressiveClassifier.predict(Y)
accuracy = accuracy_score(testLabels, pred)
confusionMatrix = confusion_matrix(testLabels, pred)
classificationReport = classification_report(testLabels, pred)
print("accuracy: ", accuracy)
print("confusion_matrix: ", confusionMatrix)
print("classification_report: ", classificationReport)