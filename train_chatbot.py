import os
import pickle
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, LSTM, Embedding, Dense, AdditiveAttention, Concatenate
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint

from utils import clean_text, tokenize_and_pad, save_tokenizer

# Load preprocessed pairs (from DailyDialog)
with open("chat_pairs.pkl", "rb") as f:
    pairs = pickle.load(f)

print(f"Loaded {len(pairs)} DailyDialog pairs.")
print("Example:", pairs[0])

# Clean and split pairs
input_texts = [clean_text(p[0]) for p in pairs]
target_texts = ['<start> ' + clean_text(p[1]) + ' <end>' for p in pairs]

# Filter long sentences
MAX_LEN = 40
filtered_inputs, filtered_targets = [], []
for inp, tgt in zip(input_texts, target_texts):
    if len(inp.split()) <= MAX_LEN and len(tgt.split()) <= MAX_LEN:
        filtered_inputs.append(inp)
        filtered_targets.append(tgt)

input_texts, target_texts = filtered_inputs, filtered_targets
print(f"After filtering: {len(input_texts)} usable pairs (<= {MAX_LEN} tokens).")


NUM_SAMPLES = min(len(input_texts), 25000)
input_texts = input_texts[:NUM_SAMPLES]
target_texts = target_texts[:NUM_SAMPLES]

print("Training samples:", len(input_texts))

# 2) Tokenize and pad
encoder_input, tokenizer_enc, max_enc_len = tokenize_and_pad(input_texts)
decoder_full, tokenizer_dec, max_dec_len = tokenize_and_pad(target_texts)

# Save tokenizers for inference
save_tokenizer(tokenizer_enc, "tokenizer_enc.pkl")
save_tokenizer(tokenizer_dec, "tokenizer_dec.pkl")

meta = {
    "max_enc_len": max_enc_len,
    "max_dec_len": max_dec_len,
    "enc_vocab": len(tokenizer_enc.word_index) + 1,
    "dec_vocab": len(tokenizer_dec.word_index) + 1
}
with open("meta.pkl", "wb") as f:
    pickle.dump(meta, f)

print("Max lengths:", max_enc_len, max_dec_len)
print("Vocab sizes:", meta["enc_vocab"], meta["dec_vocab"])

# Prepare decoder input and target
decoder_input = np.array([seq[:-1] for seq in decoder_full])
decoder_target = np.array([seq[1:] for seq in decoder_full])
decoder_target_expanded = np.expand_dims(decoder_target, -1)

#  Build model
EMBED_DIM = 128
UNITS = 256
enc_vocab = meta["enc_vocab"]
dec_vocab = meta["dec_vocab"]

enc_inputs = Input(shape=(None,), name="encoder_inputs")
enc_emb = Embedding(enc_vocab, EMBED_DIM, name="enc_embedding")(enc_inputs)
enc_lstm = LSTM(UNITS, return_sequences=True, return_state=True, name="enc_lstm")
enc_outs, enc_h, enc_c = enc_lstm(enc_emb)

dec_inputs = Input(shape=(None,), name="decoder_inputs")
dec_emb = Embedding(dec_vocab, EMBED_DIM, name="dec_embedding")(dec_inputs)
dec_lstm = LSTM(UNITS, return_sequences=True, return_state=True, name="dec_lstm")
dec_outs, _, _ = dec_lstm(dec_emb, initial_state=[enc_h, enc_c])

attn = AdditiveAttention(name="attention")
context = attn([dec_outs, enc_outs])
concat = Concatenate(axis=-1)([dec_outs, context])
output_dense = Dense(dec_vocab, activation="softmax", name="output_dense")
dec_pred = output_dense(concat)

model = Model([enc_inputs, dec_inputs], dec_pred)
model.compile(optimizer=Adam(learning_rate=1e-3), loss="sparse_categorical_crossentropy")
model.summary()

# Training setup
early_stop = EarlyStopping(monitor="val_loss", patience=3, restore_best_weights=True, verbose=1)
lr_reduce = ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=2, verbose=1, min_lr=1e-5)
checkpoint = ModelCheckpoint("chatbot_best.keras", monitor="val_loss", save_best_only=True, verbose=1)

# Train
history = model.fit(
    [encoder_input, decoder_input],
    decoder_target_expanded,
    batch_size=64,
    epochs=20,
    validation_split=0.1,
    callbacks=[early_stop, lr_reduce, checkpoint]
)

model.save("chatbot_full.keras")

# Build inference models
encoder_model = Model(enc_inputs, [enc_outs, enc_h, enc_c])
encoder_model.save("encoder_model.keras")

dec_state_input_h = Input(shape=(UNITS,), name="dec_state_h")
dec_state_input_c = Input(shape=(UNITS,), name="dec_state_c")
enc_outs_input = Input(shape=(None, UNITS), name="enc_outs_input")

dec_emb2 = model.get_layer("dec_embedding")(dec_inputs)
dec_outs2, dec_h2, dec_c2 = dec_lstm(dec_emb2, initial_state=[dec_state_input_h, dec_state_input_c])
attn_out2 = attn([dec_outs2, enc_outs_input])
concat2 = Concatenate(axis=-1)([dec_outs2, attn_out2])
dec_pred2 = output_dense(concat2)

decoder_model = Model(
    [dec_inputs, enc_outs_input, dec_state_input_h, dec_state_input_c],
    [dec_pred2, dec_h2, dec_c2]
)
decoder_model.save("decoder_model.keras")

print("Training complete. Models and tokenizers saved successfully.")
