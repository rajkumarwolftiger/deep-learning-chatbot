# Deep Learning Chatbot

A simple intent-based chatbot built with Python, TensorFlow/Keras, and
NLTK. It learns to classify user messages into predefined intent
categories and selects a response associated with the predicted intent.

## Features

-   **Intent classification:** Predicts the intent behind a user's
    message.
-   **Text preprocessing:** Uses NLTK tokenization and Lancaster
    stemming.
-   **Bag-of-words representation:** Converts text into numerical input
    features.
-   **Neural network:** Uses a Keras feed-forward neural network with
    two hidden layers.
-   **Confidence threshold:** Returns a response only when the
    prediction confidence is above the configured threshold.
-   **Model persistence:** Saves the trained model so it can be reused
    without training every time.
-   **Customizable responses:** Add or edit intents and responses in
    `intents.json`.

## Project structure

``` text
chatbot/
├── .venv/                 # Local virtual environment (do not commit)
├── app.py                 # Preprocessing, model training/loading, and chat loop
├── intents.json           # Intent tags, example patterns, and responses
├── data.pickle            # Processed vocabulary, labels, and training data
├── chatbot_model.keras    # Saved Keras model (generated after training)
├── requirements.txt       # Python dependencies
├── LICENSE                # MIT License
└── README.md              # Project documentation
```

Some generated files, such as `data.pickle` and `chatbot_model.keras`,
may be created when the application runs.

## Requirements

-   Python version compatible with the TensorFlow version in
    `requirements.txt`
-   pip

## Installation

### 1. Clone the repository

``` bash
git clone https://github.com/rajkumarwolftiger/deep-learning-chatbot.git
cd deep-learning-chatbot
```

### 2. Create and activate a virtual environment

**Windows PowerShell:**

``` powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
```

**macOS / Linux:**

``` bash
python3 -m venv .venv
source .venv/bin/activate
```

If PowerShell blocks activation, you can use Command Prompt instead:

``` cmd
.venv\Scripts\activate.bat
```

### 3. Install dependencies

``` bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

A typical `requirements.txt` for the Keras-based version is:

``` text
tensorflow==2.21.0
nltk
numpy
```

Keep `tflearn` out of the requirements: this project uses Keras, not
TFLearn. If your actual `requirements.txt` differs, make sure it matches
the imports in `app.py`.

## Run the chatbot

From the project root, with the virtual environment activated:

``` bash
python app.py
```

The application loads the intent data, prepares the training data, and
either loads the saved model or trains and saves a model if one is not
available.

When the chat prompt appears, enter a message. Type `quit` to exit.

## Customize the chatbot

Open `intents.json`. Each intent should contain:

-   `tag`: A unique name for the intent.
-   `patterns`: Example messages that should map to the intent.
-   `responses`: Possible replies the chatbot can choose from.

Example:

``` json
{
  "intents": [
    {
      "tag": "greeting",
      "patterns": ["Hello", "Hi", "Hey there"],
      "responses": ["Hello!", "Hi! How can I help?"]
    }
  ]
}
```

Add several varied patterns for each intent. A model cannot reliably
recognize examples it has never learned from if the training data is too
small or repetitive.

## How it works

1.  **Load intents:** Read the patterns, tags, and responses from
    `intents.json`.
2.  **Tokenize and stem:** Split each pattern into tokens and reduce
    words to their stems using NLTK.
3.  **Build vocabulary:** Create a sorted vocabulary of unique stemmed
    words and a sorted list of intent labels.
4.  **Create training data:** Convert each pattern into a bag-of-words
    vector and each intent into a one-hot target vector.
5.  **Train or load:** Train the Keras neural network when no saved
    model is available; otherwise load the saved model.
6.  **Predict:** Convert each user message into a bag-of-words vector
    and predict its intent.
7.  **Respond:** If the highest prediction score exceeds the confidence
    threshold, choose a response for that intent. Otherwise, ask the
    user to rephrase.

## Model and data files

-   `chatbot_model.keras` stores the trained Keras model.
-   `data.pickle` stores the vocabulary, labels, and prepared training
    arrays.

If you change the intent patterns or tags and the chatbot behaves as if
it is using old data, delete the generated `data.pickle` and
`chatbot_model.keras` files and run the application again. They will be
rebuilt. Do not delete `intents.json`.

## Limitations

-   This is a **rule-and-intent-based neural chatbot**, not a generative
    AI chatbot.
-   It can only respond to intents represented in its training data.
-   Its performance depends heavily on the quality and variety of the
    example patterns.
-   A high confidence score does not guarantee that a prediction is
    correct.
-   The current implementation is designed as a command-line
    application.

## License

This project is licensed under the MIT License. See the
[LICENSE](LICENSE) file for details.

## Author

**rajkumarwolftiger**

GitHub: [@rajkumarwolftiger](https://github.com/rajkumarwolftiger)
