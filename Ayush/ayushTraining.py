from keras.optimizers import SGD
from tensorflow.python.keras.layers import Dense, Activation, Dropout
from keras.models import Sequential
from nltk.stem import WordNetLemmatizer
import random
import pickle
import json
import numpy as np

import nltk
# nltk.download('punkt')
# nltk.download('wordnet')

lemmatizer = WordNetLemmatizer()

intents_data = json.loads(open('intent.json').read())

words = []
classes = []
documents = []

ignore_letter = ['?', '!', '.', ',']

for intent in intents_data['intents']:
    for pattern in intent['patterns']:
        word_list = nltk.word_tokenize(pattern)
        words.extend(word_list)
        documents.append((word_list, intent['name']))
        if intent['name'] not in classes:
            classes.append(intent['name'])

words = [lemmatizer.lemmatize(word)
         for word in words if word not in ignore_letter]
words = sorted(set(words))
classes = sorted(set(classes))

pickle.dump(words, open('words.pkl', 'wb'))
pickle.dump(classes, open('classes.pkl', 'wb'))

training = []
output_empty = [0]*len(classes)

for document in documents:
    bag = []
    word_patterns = document[0]
    word_patterns = [lemmatizer.lemmatize(
        word.lower()) for word in word_patterns]
    for word in words:
        bag.append(1) if word in word_patterns else bag.append(0)

    output_row = list(output_empty)
    output_row[classes.index(document[1])] = 1
    training.append([bag, output_row])

random.shuffle(training)
training = np.array(training)

train_x = list(training[:, 0])
train_y = list(training[:, 1])

model = Sequential()
model.add(Dense(128, input_shape=(len(train_x[0]),), activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(len(train_y[0]), activation='softmax'))

sgd = SGD(learning_rate=0.01, momentum=0.9, nesterov=True)

model.compile(loss='categorical_crossentropy',
              optimizer=sgd, metrics=['accuracy'])

ays = model.fit(np.array(train_x), np.array(train_y),
                epochs=200, batch_size=5, verbose=1)
model.save('Ayush_model.model', ays)
print("Done")
