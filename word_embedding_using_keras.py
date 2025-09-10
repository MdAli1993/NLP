from tensorflow.keras.layers import Embedding
from tensorflow.keras.models import Sequential
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import one_hot

sentences = ["I am a software doveloper",
            "I love programming",
            "word embedding is a good technique",
            "Today's weather is so hot",
            "Football is a nice sport"]

vocabSize = 10000
#onehot representation
oneHotRepresentation = [one_hot(words, vocabSize) for words in sentences]

maxSentenceLength = max([len(sentence.split(" ")) for sentence in sentences])
#padding
embeddedWords = pad_sequences(oneHotRepresentation, padding="pre", maxlen=maxSentenceLength)
#embedding layer
vectorFeaturesSize=10
model = Sequential()
model.add(Embedding(vocabSize, vectorFeaturesSize, input_length=maxSentenceLength))
model.compile("adam", "mse")
model.summary()

print(oneHotRepresentation[0])
print(embeddedWords[0])
print(model.predict(embeddedWords)[0])