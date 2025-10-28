# Evaluate_chatbot.py
import numpy as np
import pandas as pd
import random
from tqdm import tqdm
import re
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
from rouge import Rouge
from bert_score import score as bert_score
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
import pickle
from utils import load_tokenizer, clean_text

# -----------------------------
# Load models and tokenizers
# -----------------------------
print("🔹 Loading models and tokenizers...")
encoder_model = load_model("encoder_model.keras", compile=False)
decoder_model = load_model("decoder_model.keras", compile=False)
tokenizer_enc = load_tokenizer("tokenizer_enc.pkl")
tokenizer_dec = load_tokenizer("tokenizer_dec.pkl")

with open("meta.pkl", "rb") as f:
    meta = pickle.load(f)

MAX_LEN_INPUT = meta["max_enc_len"]
MAX_LEN_TARGET = meta["max_dec_len"]
reverse_dec_index = {idx: word for word, idx in tokenizer_dec.word_index.items()}

# -----------------------------
# Decode with temperature sampling
# -----------------------------
def sample_with_temperature(preds, temperature=0.8, top_k=30):
    preds = np.asarray(preds).astype("float64")
    preds = np.log(preds + 1e-8) / temperature
    exp_preds = np.exp(preds)
    preds = exp_preds / np.sum(exp_preds)
    top_k = min(top_k, len(preds))
    top_indices = preds.argsort()[-top_k:]
    top_probs = preds[top_indices]
    top_probs /= np.sum(top_probs)
    return np.random.choice(top_indices, p=top_probs)

def decode_sequence(input_text):
    seq = tokenizer_enc.texts_to_sequences([clean_text(input_text)])
    enc_input = pad_sequences(seq, maxlen=MAX_LEN_INPUT, padding='post')
    enc_outs, state_h, state_c = encoder_model.predict(enc_input)
    start_token = tokenizer_dec.word_index.get('<start>')
    end_token = tokenizer_dec.word_index.get('<end>')
    target_seq = np.array([[start_token]])

    decoded_sentence = []
    for _ in range(MAX_LEN_TARGET):
        preds, h, c = decoder_model.predict([target_seq, enc_outs, state_h, state_c])
        preds = preds[0, -1, :]
        sampled_token_index = sample_with_temperature(preds, 0.8, 30)
        sampled_word = reverse_dec_index.get(sampled_token_index, "")
        if sampled_word == '<end>' or sampled_word == "":
            break
        decoded_sentence.append(sampled_word)
        target_seq = np.array([[sampled_token_index]])
        state_h, state_c = h, c

    return " ".join(decoded_sentence).strip().capitalize()

# -----------------------------
# Load validation data
# -----------------------------
print("\n📂 Loading validation data...")
df = pd.read_csv("validation.csv")
pairs = []

def extract_dialogue(text):
    return re.findall(r"['\"](.*?)['\"]", text)

for _, row in df.iterrows():
    text = row["dialog"]
    lines = extract_dialogue(text)
    if len(lines) > 1:
        for j in range(len(lines) - 1):
            pairs.append((lines[j].strip(), lines[j + 1].strip()))

print(f"✅ Loaded {len(pairs)} validation pairs.\n")

# -----------------------------
# Evaluate the chatbot
# -----------------------------
if not pairs:
    print("❌ No usable pairs found. Please check validation.csv formatting.")
    exit()

# Filter out empty responses
filtered_pairs = [(q, a) for (q, a) in pairs if q.strip() and a.strip()]
print(f"✅ Using {len(filtered_pairs)} clean pairs after filtering.\n")

print("Evaluating chatbot on a random subset of the validation set...\n")
sampled_pairs = random.sample(filtered_pairs, min(100, len(filtered_pairs)))

refs, hyps = [], []
rouge = Rouge()
smoothie = SmoothingFunction().method4

for q, ref in tqdm(sampled_pairs, desc="Evaluating"):
    pred = decode_sequence(q)
    refs.append(ref.strip())
    hyps.append(pred.strip())

# Remove any remaining empty items before scoring
clean_data = [(r, h) for r, h in zip(refs, hyps) if r and h]
refs, hyps = zip(*clean_data)

# BLEU Scores
bleu_scores = [sentence_bleu([r.split()], h.split(), smoothing_function=smoothie) for r, h in zip(refs, hyps)]
avg_bleu = np.mean(bleu_scores)

# ROUGE Scores
rouge_scores = rouge.get_scores(hyps, refs, avg=True)

# BERTScore
P, R, F1 = bert_score(hyps, refs, lang="en", verbose=True)

print("\n📊 Evaluation Metrics:")
print(f"BLEU Score: {avg_bleu:.4f}")
print(f"ROUGE-1: {rouge_scores['rouge-1']['f']:.4f}")
print(f"ROUGE-L: {rouge_scores['rouge-l']['f']:.4f}")
print(f"BERTScore (F1): {F1.mean().item():.4f}")

print("\n✅ Evaluation complete. Metrics computed on", len(refs), "valid dialogue pairs.")
