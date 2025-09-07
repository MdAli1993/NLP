import nltk
import pandas as pd
from nltk.corpus import stopwords
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import (accuracy_score, classification_report,
                             confusion_matrix)

df = pd.read_csv("data/stock_headlines_with_labels.csv", encoding="latin1")

# Train-Test split
splitIndex = int(len(df) * 0.9)
trainDf = df[:splitIndex]
testDf = df[splitIndex:]


# Split into data and labels
trainData = trainDf.iloc[:, 2:]
trainLabel = trainDf.iloc[:, 1]
testData = testDf.iloc[:, 2:]
testLabel = testDf.iloc[:,1]

# Filter the numerical and special characters.
trainData.replace("[^a-zA-Z]", " ", regex=True, inplace=True)
testData.replace("[^a-zA-Z]", " ", regex=True, inplace=True)

# convert the characters into lowercase, and merge all the column values per rows.
for columnName in trainData.columns:
    trainData[columnName] = trainData[columnName].str.lower()
trainDataMergedRows = []
for _, row in trainData.iterrows():
    trainDataMergedRows.append(" ".join(str(sentence) for sentence in row))

# BagOfWords. use the CountVectorizer with bigrams as to get 2-word sequences
countVectorizer = CountVectorizer(ngram_range=(2,2))
trainDataset = countVectorizer.fit_transform(trainDataMergedRows)

# Model training with RandomForest classifier
randomClassifier = RandomForestClassifier(n_estimators=200, criterion="entropy")
randomClassifier.fit(trainDataset, trainLabel)

# convert the characters into lowercase, and merge all the column values per rows.
for columnName in testData.columns:
    testData[columnName] = testData[columnName].str.lower()
testDataMergedRows = []
for _, row in testData.iterrows():
    testDataMergedRows.append(" ".join(str(sentence) for sentence in row))
testDataset = countVectorizer.transform(testDataMergedRows)

# Predict on test data
y_pred = randomClassifier.predict(testDataset)
accuracy = accuracy_score(testLabel, y_pred)
confusionMatrix = confusion_matrix(testLabel, y_pred)
classificationReport = classification_report(testLabel, y_pred)
print("accuracy: ", accuracy)
print("confusion matrix: ", confusionMatrix)
print("classification report: ", classificationReport)