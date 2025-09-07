import re

import matplotlib.pyplot as plt
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer, WordNetLemmatizer
from nltk.tokenize import sent_tokenize, word_tokenize
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

# nltk.download(download_dir="../NLP_venv/nltk_data")
# nltk.data.path.insert(0, "../NLP_venv/nltk_data")

paragraph = """Tokenization in Natural Language Processing (NLP) is the process of breaking down text into smaller, meaningful units called tokens.
These tokens can be as small as characters, as large as sentences, but most often they are words or subwords.
Tokenization is a crucial first step in many NLP tasks because computers process text as structured data rather than free-flowing language.
Word-level tokenization is intuitive but struggles with rare or unknown words, which is why modern NLP models often rely on subword tokenization techniques such as Byte Pair Encoding (BPE) or WordPiece.
By transforming text into tokens, NLP systems can better analyze, interpret, and model human language for applications like translation, chatbots, and search engines."""

sentences = sent_tokenize(paragraph)
stemmer = PorterStemmer()
lemmatizer = WordNetLemmatizer()

stemmedSentences = []
lemmatizedSentences = []

for idx in range(len(sentences)):
    cleanedWords = re.sub("[^a-zA-Z]", " ", sentences[idx])
    words= word_tokenize(cleanedWords)
    # Stemming
    stemmedWords = [stemmer.stem(word.lower()) for word in words if word.lower() not in set(stopwords.words("english"))]
    stemmedSentences.append(" ".join(stemmedWords))
    # Lemmatization
    lemmatizedWords = [lemmatizer.lemmatize(word.lower()) for word in words if word.lower() not in set(stopwords.words("english"))]
    lemmatizedSentences.append(" ".join(lemmatizedWords))

print("List of sentences after stemming: ", stemmedSentences)
print("List of sentences after lemmatization: ", lemmatizedSentences)

# Bag Of Words
countVectorizeBOW = CountVectorizer(max_features= 1500)
bagOfWords = countVectorizeBOW.fit_transform(lemmatizedSentences).toarray()

# TF-IDF
countVectorizeTFIDF = TfidfVectorizer()
tfidf = countVectorizeTFIDF.fit_transform(lemmatizedSentences).toarray()