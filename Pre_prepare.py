import re
import pickle
import pandas as pd
import ast

def clean_text(text):
    """Clean and normalize text."""
    text = re.sub(r"[^a-zA-Z0-9?.,!']+", " ", text)
    return text.strip().lower()

def parse_dialogue(text):
    """Parse DailyDialog 'dialog' field safely."""
    # Normalizing quotes and spacing
    text = text.replace("“", '"').replace("”", '"').replace("’", "'")
    text = text.replace("`", "'").replace("´", "'")

    # Fixing missing commas between utterances: insert comma between ']'' and '['
    text = text.replace("' '", "', '")

    try:
        parsed = ast.literal_eval(text)
        if isinstance(parsed, list):
            return [clean_text(u) for u in parsed if isinstance(u, str)]
    except Exception as e:
        # fallback: try splitting manually
        utterances = re.findall(r"'(.*?)'", text)
        return [clean_text(u) for u in utterances if len(u.strip()) > 0]

    return []

def load_dailydialog_csv(filename):
    """Load DailyDialog CSV and extract conversation pairs."""
    df = pd.read_csv(filename)
    pairs = []

    for dialogue_str in df["dialog"]:
        utterances = parse_dialogue(dialogue_str)
        for i in range(len(utterances) - 1):
            if utterances[i] and utterances[i + 1]:
                pairs.append((utterances[i], utterances[i + 1]))
    return pairs

if __name__ == "__main__":
    print("🔹 Loading DailyDialog dataset...")

    train_pairs = load_dailydialog_csv("train.csv")
    print(f"Loaded {len(train_pairs)} training pairs.")

    if len(train_pairs) > 0:
        print("\n🔹 Example conversation pairs:")
        for i in range(min(3, len(train_pairs))):
            print(train_pairs[i])
    else:
        print("No pairs found. Check parsing logic or CSV format.")

    # Save pairs for training
    with open("chat_pairs.pkl", "wb") as f:
        pickle.dump(train_pairs, f)

    print("\n Saved conversation pairs to chat_pairs.pkl")
