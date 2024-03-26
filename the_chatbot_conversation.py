import random
from keras.models import load_model
import json
import pickle
import numpy as np
from nltk.stem import WordNetLemmatizer
import nltk

# Load trained model and associated files
# Make sure to use your paths...
model = load_model("chatbotmodel.h5")
intents = json.loads(open("intents.json").read())
words = pickle.load(open('words.pkl', 'rb'))
classes = pickle.load(open('classes.pkl', 'rb'))

lemmatizer = WordNetLemmatizer()


def clean_up_sentence(sentence):
    sentence_words = nltk.word_tokenize(sentence)
    sentence_words = [lemmatizer.lemmatize(
        word.lower()) for word in sentence_words]
    return sentence_words


def bow(sentence, words, show_details=True):
    sentence_words = clean_up_sentence(sentence)
    bag = [0]*len(words)
    for s in sentence_words:
        for i, w in enumerate(words):
            if w == s:
                bag[i] = 1
                if show_details:
                    print("found in bag: %s" % w)
    return (np.array(bag))


def predict_class(sentence):
    p = bow(sentence, words, show_details=False)
    res = model.predict(np.array([p]))[0]
    ERROR_THRESHOLD = 0.25
    results = [[i, r] for i, r in enumerate(res) if r > ERROR_THRESHOLD]
    results.sort(key=lambda x: x[1], reverse=True)
    return_list = []
    for r in results:
        return_list.append({"intent": classes[r[0]], "probability": str(r[1])})
    return return_list


def get_response(ints, intents_json):
    tag = ints[0]['intent']
    list_of_intents = intents_json['intents']
    for i in list_of_intents:
        if (i['tag'] == tag):
            result = random.choice(i['responses'])
            break
    return result


# Main conversation loop
print("Welcome! How can I assist you today?")
while True:
    user_input = input("You: ")
    if user_input.lower() == 'quit':
        break

    # Predict the intent of the user input
    ints = predict_class(user_input)

    # Get a response based on the predicted intent
    response = get_response(ints, intents)
    print("Bot:", response)
