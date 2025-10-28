# Inference.py
import numpy as np
import random
import pickle
import os
import csv
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
from utils import load_tokenizer, preprocess_input, clean_text

print(" Loading models and tokenizers...")

# Load models and metadata
encoder_model = load_model("encoder_model.keras", compile=False)
decoder_model = load_model("decoder_model.keras", compile=False)

tokenizer_enc = load_tokenizer("tokenizer_enc.pkl")
tokenizer_dec = load_tokenizer("tokenizer_dec.pkl")

with open("meta.pkl", "rb") as f:
    meta = pickle.load(f)

MAX_LEN_INPUT = meta["max_enc_len"]
MAX_LEN_TARGET = meta["max_dec_len"]

# Reverse index mapping for decoder
reverse_dec_index = {idx: word for word, idx in tokenizer_dec.word_index.items()}

# Temperature-based sampling
def sample_with_temperature(preds, temperature=0.8, top_k=30):
    preds = np.asarray(preds).astype("float64")
    preds = np.log(preds + 1e-8) / temperature
    exp_preds = np.exp(preds)
    preds = exp_preds / np.sum(exp_preds)

    # top-k filtering
    top_k = min(top_k, len(preds))
    top_indices = preds.argsort()[-top_k:]
    top_probs = preds[top_indices]
    top_probs /= np.sum(top_probs)
    return np.random.choice(top_indices, p=top_probs)

# Decode function (beam-like sampling)
def decode_sequence(input_text):
    input_text = clean_text(input_text)
    seq = tokenizer_enc.texts_to_sequences([input_text])
    enc_input = pad_sequences(seq, maxlen=MAX_LEN_INPUT, padding='post')

    enc_outs, state_h, state_c = encoder_model.predict(enc_input)

    start_token = tokenizer_dec.word_index.get('<start>')
    end_token = tokenizer_dec.word_index.get('<end>')
    target_seq = np.array([[start_token]])

    decoded_sentence = []
    for _ in range(MAX_LEN_TARGET):
        preds, h, c = decoder_model.predict([target_seq, enc_outs, state_h, state_c])
        preds = preds[0, -1, :]

        # Temperature + top-k sampling
        sampled_token_index = sample_with_temperature(preds, temperature=0.8, top_k=30)
        sampled_word = reverse_dec_index.get(sampled_token_index, "")

        if sampled_word == '<end>' or sampled_word == "":
            break

        decoded_sentence.append(sampled_word)
        target_seq = np.array([[sampled_token_index]])
        state_h, state_c = h, c

    return " ".join(decoded_sentence).strip().capitalize()

# Conversation Logger
LOG_FILE = "chat_logs.csv"

def log_conversation(user_text, bot_response):
    # Create the log file with a header if it doesn’t exist
    if not os.path.exists(LOG_FILE):
        with open(LOG_FILE, mode='w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(["user_input", "bot_response"])

    # Append new conversation to the file
    with open(LOG_FILE, mode='a', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow([user_text, bot_response])

# Chat loop
print("\n Chatbot is ready! Type 'quit' to exit.\n")

while True:
    user_input = input("You: ").strip()
    if user_input.lower() in ["quit", "exit", "bye"]:
        print("Bot: Goodbye! 👋")
        break
    response = decode_sequence(user_input)
    print("Bot:", response, "\n")

    # Log each conversation to CSV
    log_conversation(user_input, response)
