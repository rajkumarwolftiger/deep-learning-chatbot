import nltk
from nltk.stem.lancaster import LancasterStemmer

import numpy as np
import tensorflow as tf
from tensorflow import keras

import random
import pickle
import json
import os
stemmer = LancasterStemmer()

with open("intents.json", "r", encoding="utf-8") as f:
    data = json.load(f)

try:
    nltk.word_tokenize("test")
except LookupError:
    nltk.download("punkt")
    nltk.download("punkt_tab")

if os.path.exists("data.pickle"):
    with open("data.pickle", "rb") as m:
        words, labels, training, output = pickle.load(m)

else:
    words = []
    labels = []
    docs_x = []
    docs_y = []

    for intent in data["intents"]:
        for pattern in intent["patterns"]:
            wrds = nltk.word_tokenize(pattern)

            words.extend(wrds)
            docs_x.append(wrds)
            docs_y.append(intent["tag"])

        if intent["tag"] not in labels:
            labels.append(intent["tag"])

    words = [
        stemmer.stem(w.lower())
        for w in words
        if w != "?"
    ]

    words = sorted(list(set(words)))
    labels = sorted(labels)

    training = []
    output = []

    out_empty = [0 for _ in range(len(labels))]

    for x, doc in enumerate(docs_x):
        bag = []

        wrds = [
            stemmer.stem(w.lower())
            for w in doc
        ]

        for w in words:
            bag.append(1 if w in wrds else 0)

        output_row = out_empty[:]
        output_row[labels.index(docs_y[x])] = 1

        training.append(bag)
        output.append(output_row)

    training = np.array(training, dtype=np.float32)
    output = np.array(output, dtype=np.float32)

    with open("data.pickle", "wb") as m:
        pickle.dump(
            (words, labels, training, output),
            m
        )

training = np.asarray(training, dtype=np.float32)
output = np.asarray(output, dtype=np.float32)


model_path = "chatbot_model.keras"

model = keras.Sequential([
    keras.Input(shape=(len(training[0]),)),

    keras.layers.Dense(8, activation="relu"),

    keras.layers.Dense(8, activation="relu"),

    keras.layers.Dense(
        len(output[0]),
        activation="softmax"
    )
])

model.compile(
    optimizer="adam",
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)

if os.path.exists(model_path):
    print("Loading existing chatbot model...")
    model = keras.models.load_model(model_path)

else:
    print("Training chatbot model...")

    model.fit(
        training,
        output,
        epochs=1000,
        batch_size=8,
        verbose=1
    )

    model.save(model_path)

    print("Model saved successfully!")

def bag_of_words(s, words):
    bag = [0 for _ in range(len(words))]

    s_words = nltk.word_tokenize(s)

    s_words = [
        stemmer.stem(word.lower())
        for word in s_words
    ]

    for se in s_words:
        for i, w in enumerate(words):
            if w == se:
                bag[i] = 1

    return np.array(bag, dtype=np.float32)


def chat():

    print("\n===================================")
    print(" Chatbot is ready!")
    print(" Type 'quit' to exit.")
    print("===================================\n")

    while True:

        inp = input("You: ").strip()

        if inp.lower() == "quit":
            print("Bot: Goodbye!")
            break

        if not inp:
            continue

        input_bag = bag_of_words(inp, words)

        results = model.predict(
            np.array([input_bag]),
            verbose=0
        )[0]

        results_index = np.argmax(results)

        tag = labels[results_index]

        confidence = results[results_index]

        if confidence > 0.7:

            responses = []

            for tg in data["intents"]:
                if tg["tag"] == tag:
                    responses = tg["responses"]
                    break

            if responses:
                print("Bot:", random.choice(responses))
            else:
                print("Bot: I couldn't find a response.")

        else:
            print("Bot: I didn't get that, try again.")


if __name__ == "__main__":
    chat()
