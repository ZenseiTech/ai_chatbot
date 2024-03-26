# import
from nltk.corpus import stopwords
from keras.optimizers import SGD
from keras.layers import Dense, Activation, Dropout
from nltk.stem import WordNetLemmatizer
from keras.models import Sequential
import random
import json
import pickle
import numpy as np
import nltk
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('stopwords')

# Reading the JSON file. Make sure to use your path ...
intents = json.loads(open("intents.json").read())

# Creating empty lists to store data
simple_words = []
classes = []
documents = []

for intent in intents['intents']:
    for pattern in intent['patterns']:
        # tokenizing the pattern ...
        word_list = nltk.word_tokenize(pattern)
        simple_words.extend(word_list)

        # Associate the tokenize pattern list with respective tags
        documents.append((word_list, intent['tag']))

        # append tags to class list
        if intent['tag'] not in classes:
            classes.append(intent['tag'])

# Storing the root words or lemma
all_stopwords = stopwords.words('english')

# change plural words to singular ...
lemmatizer = WordNetLemmatizer()
lemmatize_words = [lemmatizer.lemmatize(simple_word)
                   for simple_word in simple_words if simple_word not in set(all_stopwords)]
lemmatize_words = sorted(set(lemmatize_words))

# Training
training = []
output_empty = [0]*len(classes)
for document in documents:
    bag = []
    word_patterns = document[0]
    lemmatize_word_patterns = [lemmatizer.lemmatize(
        word.lower()) for word in word_patterns]

    for lemmatize_word in lemmatize_words:
        bag.append(
            1) if lemmatize_word in lemmatize_word_patterns else bag.append(0)

    # Make a copy of outputEmpty
    output_row = list(output_empty)
    output_row[classes.index(document[1])] = 1

    bag.extend(output_row)
    training.append(bag)

random.shuffle(training)
training = np.array(training)

# Split data
trainX = training[:, :len(lemmatize_words)]
trainY = training[:, len(lemmatize_words):]

# Create sequential machine learning model
model = Sequential()

# adding deep learning layers....
model.add(Dense(128, input_shape=(len(trainX[0]),), activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.5))
# Corrected output layer
model.add(Dense(len(classes), activation='softmax'))

# Compile model
sgd = SGD(learning_rate=0.01, momentum=0.9, nesterov=True)
model.compile(loss='categorical_crossentropy',
              optimizer=sgd, metrics=['accuracy'])

# start the training ...
hist = model.fit(trainX, trainY, epochs=200, batch_size=5, verbose=True)

# Save words and class list to binary files, to be loaded later
#  when creating conversation ....
pickle.dump(lemmatize_words, open('words.pkl', 'wb'))
pickle.dump(classes, open('classes.pkl', 'wb'))

# Save Model, to be used later for conversation ...
model.save("chatbotmodel.h5", hist)

print("Training is finished!")
