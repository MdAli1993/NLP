import re

import nltk
import pandas as pd
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import (accuracy_score, classification_report,
                             confusion_matrix)
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB

# nltk.download("stopwords")

messages = pd.read_csv("data/sms_spam_collection/SMSSpamCollection", sep="\t", names=["label", "message"])


stemmer = PorterStemmer()
stemmedSentences = []
stopWords = set(stopwords.words("english"))
for i in range(len(messages)):
    cleanedWords = re.sub("[^a-zA-Z]", " ", messages["message"][i])
    words = word_tokenize(cleanedWords)
    stemmedWords = [stemmer.stem(word.lower()) for word in words if word not in stopWords]
    stemmedSentences.append(" ".join(stemmedWords))

# BagOfWords
countVectorizer = CountVectorizer(max_features=2500) # take top 2500 most frequent words
X = countVectorizer.fit_transform(stemmedSentences).toarray()


y = pd.get_dummies(messages["label"]) # convert the actual label values into dummy variable
y = y.iloc[:,1].values # reduce into one categorical variable column i.e, spam column (ham:0, spam:1)

# Train-Test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=21)

# Training the model using Naive bayes classifier
spamDetector = MultinomialNB().fit(X_train, y_train)

# Predict the model on test data
y_pred = spamDetector.predict(X_test)


# confusionMatrix = confusion_matrix(y_test, y_pred)
accuracy = accuracy_score(y_test, y_pred)
print("accuracy", accuracy)
print("classification report: ",classification_report(y_test, y_pred))