# utils.py

import re
import pickle
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

# ==============================
# Text Cleaning
# ==============================
def clean_text(text):
    text = text.lower()
    text = re.sub(r"i'm", "i am", text)
    text = re.sub(r"he's", "he is", text)
    text = re.sub(r"she's", "she is", text)
    text = re.sub(r"it's", "it is", text)
    text = re.sub(r"that's", "that is", text)
    text = re.sub(r"what's", "what is", text)
    text = re.sub(r"where's", "where is", text)
    text = re.sub(r"how's", "how is", text)
    text = re.sub(r"'ll", " will", text)
    text = re.sub(r"'ve", " have", text)
    text = re.sub(r"'re", " are", text)
    text = re.sub(r"'d", " would", text)
    text = re.sub(r"n't", " not", text)
    text = re.sub(r'[-()\"#/@;:<>{}`+=~|.!?,]', '', text)
    return text.strip()

# ==============================
# Tokenization + Padding
# ==============================
def tokenize_and_pad(texts, filters='', oov_token='<out>'):
    tokenizer = Tokenizer(filters=filters, oov_token=oov_token)
    tokenizer.fit_on_texts(texts)
    seq = tokenizer.texts_to_sequences(texts)
    max_len = max(len(s) for s in seq)
    padded = pad_sequences(seq, maxlen=max_len, padding='post')
    return padded, tokenizer, max_len

# ==============================
# Tokenizer Saving/Loading
# ==============================
def save_tokenizer(tokenizer, filename):
    with open(filename, 'wb') as f:
        pickle.dump(tokenizer, f)

def load_tokenizer(filename):
    with open(filename, 'rb') as f:
        return pickle.load(f)

# ==============================
# Preprocessing for Inference
# ==============================
def preprocess_input(text, tokenizer, max_len):
    seq = tokenizer.texts_to_sequences([clean_text(text)])
    return pad_sequences(seq, maxlen=max_len, padding='post')
