import re

import nltk
from gensim.models import Word2Vec
from nltk.corpus import stopwords
from nltk.tokenize import sent_tokenize, word_tokenize

paragraph = """Tokenization in Natural Language Processing (NLP) is the process of breaking down text into smaller, meaningful units called tokens.
These tokens can be as small as characters, as large as sentences, but most often they are words or subwords.
Tokenization is a crucial first step in many NLP tasks because computers process text as structured data rather than free-flowing language.
Word-level tokenization is intuitive but struggles with rare or unknown words, which is why modern NLP models often rely on subword tokenization techniques such as Byte Pair Encoding (BPE) or WordPiece.
By transforming text into tokens, NLP systems can better analyze, interpret, and model human language for applications like translation, chatbots, and search engines."""


cleanedParagraph = re.sub("[^a-zA-Z]", " ", paragraph.lower())
sentences = sent_tokenize(cleanedParagraph)
words = [word_tokenize(sentence) for sentence in sentences]

cleanedWords = []

for i in range(len(words)):
    cleanedWords.append([word for word in words[i] if word not in stopwords.words("english")])

# Training the word2vec model
model = Word2Vec(cleanedWords, min_count=1)

vocabs = model.wv.key_to_index
print("vocabs", vocabs)
# Finding vectors for a given word
vectors = model.wv["language"]
# Finding similar words to a given word.
similarWords = model.wv.most_similar("language")
print("similar words: ", similarWords)
