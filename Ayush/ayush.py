import random
import time
import json
import pickle
import numpy as np
import nltk
from nltk.stem import WordNetLemmatizer
# from flask import Flask, render_template, request
from keras.models import load_model


import pyttsx3

engine = pyttsx3.init()


lemmatizer = WordNetLemmatizer()
intent = json.loads(open('intent.json').read())

words = pickle.load(open('words.pkl', 'rb'))
classes = pickle.load(open('classes.pkl', 'rb'))
model = load_model('Ayush_model.model')  # Corrected the model file name


def clean_up_sentence(sentence):
    sentence_words = nltk.word_tokenize(sentence)  # Corrected variable name
    sentence_words = [lemmatizer.lemmatize(word) for word in sentence_words]
    return sentence_words


def bag_of_words(sentence):
    sentence_words = clean_up_sentence(sentence)
    bag = [0] * len(words)
    for w in sentence_words:
        for i, word in enumerate(words):
            if word == w:
                bag[i] = 1
    return np.array(bag)


def predict_class(sentence):
    bow = bag_of_words(sentence)
    res = model.predict(np.array([bow]))[0]
    ERROR_THRESHOLD = 0.25
    results = [[i, r] for i, r in enumerate(res) if r > ERROR_THRESHOLD]

    results.sort(key=lambda x: x[1], reverse=True)
    return_list = []
    for r in results:
        # Changed 'intents' to 'intent'
        return_list.append({'intent': classes[r[0]], 'probability': str(r[1])})
    return return_list


def get_response(intents_list, intents_json):
    intent = intents_list[0]['intent']  # Corrected variable name
    list_of_intents = intents_json['intents']
    for intent_data in list_of_intents:
        if intent_data['name'] == intent:
            result = random.choice(intent_data['responses'])

            break
    return result


print("Hey! I'm Your Virtual Friend Ayush")

while True:
    message = input("---> ")
    if message.lower() == 'quit':
        break
    intents_list = predict_class(message)
    response = get_response(intents_list, intent)
    engine.say(response)
    engine.runAndWait()
    print(response)
