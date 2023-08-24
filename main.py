import json
import numpy as np
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense

# add your .json dataset file here
with open("split_objects.json", "r", encoding="utf-8") as json_file:
    data = json.load(json_file)

# which row will be trained, you can select here.
contents = [entry["content"] for entry in data]

# tokenization and padding
max_sequence_length = 100  # You can adjust this based on your data
tokenizer = Tokenizer()
tokenizer.fit_on_texts(contents)
sequences = tokenizer.texts_to_sequences(contents)
padded_sequences = pad_sequences(sequences, maxlen=max_sequence_length, padding="post", truncating="post")

# Prepare inputs and targets
input_sequences = padded_sequences[:, :-1]
target_sequences = padded_sequences[:, 1:]

# create a simple LSTM text generation model
vocab_size = len(tokenizer.word_index) + 1
embedding_dim = 128
model = Sequential([
    Embedding(vocab_size, embedding_dim, input_length=max_sequence_length - 1),
    LSTM(128, return_sequences=True),
    Dense(vocab_size, activation="softmax")
])

model.compile(loss="sparse_categorical_crossentropy", optimizer="adam")

# train the model, (epochs) stand for how many times you need to train the machine, you can adjust it.
model.fit(input_sequences, target_sequences, epochs=20)

# save the model weights so we can use it again
model.save_weights("text_generation_model_weights.h5")

# generate text
seed_text = "hi"
seed_sequence = tokenizer.texts_to_sequences([seed_text])[0]
for _ in range(10):
    padded_seed = pad_sequences([seed_sequence], maxlen=max_sequence_length - 1, padding="post")
    predicted_probs = model.predict(padded_seed, verbose=0)[0]

    predicted_token = np.argmax(predicted_probs)
    if predicted_token >= vocab_size:
        predicted_token = np.random.choice(vocab_size)

    if predicted_token not in tokenizer.index_word:
        predicted_word = "<UNK>"
    else:
        predicted_word = tokenizer.index_word[predicted_token]

    seed_text += " " + predicted_word
    seed_sequence = seed_sequence + [predicted_token]
    seed_sequence = seed_sequence[1:]

print("Generated Text:", seed_text)
