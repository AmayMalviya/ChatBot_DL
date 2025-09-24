import os
import re
import unicodedata
import random
import pickle

# PAD is for making all my sentences the same length
# SOS is for the start of a sentence, and EOS is for the end
PAD_token = 0
SOS_token = 1
EOS_token = 2

# This is dictionary class to keep track of all the words found.

class Voc:
    def __init__(self, name):
        self.name = name
        self.trimmed = False # for later use (alrerady trimmed hai)
        
        # These are my word-to-number lookups
        self.word2index = {}
        self.word2count = {}
        self.index2word = {PAD_token: "PAD", SOS_token: "SOS", EOS_token: "EOS"}
        self.num_words = 3

    def addSentence(self, sentence):
        #to add all the words from one sentence
        for word in sentence.split(' '):
            self.addWord(word)

    def addWord(self, word):
        # adding a single word to my dictionary
        if word not in self.word2index:
            # If a word is not yet encountered a new number is assigmed hre
            self.word2index[word] = self.num_words
            self.word2count[word] = 1
            self.index2word[self.num_words] = word
            self.num_words += 1
        else:
            # If seent before count it again
            self.word2count[word] += 1

    def trim(self, min_count):
        #remove words with less occurence
        if self.trimmed:
            return

        keep_words = []
        for word, count in self.word2count.items():
            if count >= min_count:
                keep_words.append(word)

        print(f"Okay, I'm keeping {len(keep_words)} words that show up a lot.")

        #starting  dictionary over with only good words
        self.word2index = {}
        self.word2count = {}
        self.index2word = {PAD_token: "PAD", SOS_token: "SOS", EOS_token: "EOS"}
        self.num_words = 3

        for word in keep_words:
            self.addWord(word)
        
        self.trimmed = True

def unicodeToAscii(s):
    #cpnverting all to english letters
    return ''.join(
        c for c in unicodedata.normalize('NFD', s)
        if unicodedata.category(c) != 'Mn'
    )

def normalizeString(s):
    # This function cleans up sentences so theyreall consistent
    s = unicodeToAscii(s.lower().strip())
    #adding a space before .!? so it becomes one word
    s = re.sub(r"([.!?])", r" \1", s)
    #removing special characters if not grammer related
    s = re.sub(r"[^a-zA-Z.!?]+", r" ", s)
    #cleaning  any extra space
    return s.strip()

def readVocs(data_path, corpus_name):
    #reading all the movie lines and conversations.
    print("Reading lines from the files...")
    
    #checking to make sure the files are actually where I expect them to be
    lines_file_path = os.path.join(data_path, 'movie_lines.txt')
    conversations_file_path = os.path.join(data_path, 'movie_conversations.txt')
    
    if not os.path.exists(lines_file_path) or not os.path.exists(conversations_file_path):
        # If can't find the files,stop and le user know
        raise FileNotFoundError(f"couldn't find the movie files in the '{data_path}' folder. Please put them there!")
    
    #making a  dictionary to lonk each lines ID to its text
    lines = {}
    with open(lines_file_path, 'r', encoding='iso-8859-1') as f:
        for line in f:
            parts = line.strip().split(' +++$+++ ')
            if len(parts) == 5:
                lines[parts[0]] = parts[4]

    #reading the conversations file to get the list of line IDs for each chat.
    conversations = []
    with open(conversations_file_path, 'r', encoding='iso-8859-1') as f:
        for line in f:
            parts = line.strip().split(' +++$+++ ')
            if len(parts) == 4:
                #making list
                id_string = parts[3].replace("'", "").replace("[", "").replace("]", "")
                line_ids = id_string.split(", ")
                conversations.append(line_ids)

    # pairing u aech question with its anwer
    qa_pairs = []
    for conversation in conversations:
        if len(conversation) > 1:
            for i in range(len(conversation) - 1):
                input_line_id = conversation[i]
                target_line_id = conversation[i+1]
                #making sure both lines actually exist in dictionary
                if input_line_id in lines and target_line_id in lines:
                    qa_pairs.append([lines[input_line_id], lines[target_line_id]])

    #creating vocabulary object and semding bAck the pairs
    voc = Voc(corpus_name)
    return voc, qa_pairs

def filter_pairs(pairs, max_length):
    # This fumction is just to keep the sentences that are not tpp long
    filtered_list = []
    for pair in pairs:
        if len(pair[0].split(' ')) < max_length and len(pair[1].split(' ')) < max_length:
            filtered_list.append(pair)
    return filtered_list

def main():
    corpus_name = "cornell-movie-dialogs-corpus"
    MAX_LENGTH = 10  # This is the max number of words in a sentence we nee dto keep

    print("Starting the data cleanup!")

    # First, I'll read and get all the pairs of sentences.
    voc, pairs = readVocs('Data', corpus_name)
    print(f"read {len(pairs)} question-answer pairs.")

    # clean up all the text.
    normalized_pairs = []
    for pair in pairs:
        normalized_pairs.append([normalizeString(pair[0]), normalizeString(pair[1])])
    print(f"cleaned up all the pairs.")

    #filter out the long sentences.
    filtered_pairs = filter_pairs(normalized_pairs, MAX_LENGTH)
    print(f"filtered  down to {len(filtered_pairs)} good pairs.")

    #building vocabulary from all the filtered sentences.
    for pair in filtered_pairs:
        voc.addSentence(pair[0])
        voc.addSentence(pair[1])
    print("count of words")
    print(f"vocabulary size is: {voc.num_words}")

    # Now removing works seem only a few times in the dataset
    MIN_COUNT = 3
    voc.trim(MIN_COUNT)

    # to re-filter  pairs now that i have removed some workds
    keep_pairs = []
    for pair in filtered_pairs:
        input_sentence = pair[0]
        output_sentence = pair[1]
        keep_pair = True
        
        # Checking if all words in the input sentence are in vocabulary
        for word in input_sentence.split(' '):
            if word not in voc.word2index:
                keep_pair = False
                break
        
        # Checking if all words in the output sentence are in vocabulary
        if keep_pair: # Only check the output if the input was good
            for word in output_sentence.split(' '):
                if word not in voc.word2index:
                    keep_pair = False
                    break
        
        if keep_pair:
            keep_pairs.append(pair)

    print(f" final count of good pairs {len(keep_pairs)}.")
    
    random.shuffle(keep_pairs) #shuffling to make it more random for training.

    # saving the data
    with open('qa_pairs.pkl', 'wb') as f:
        pickle.dump(keep_pairs, f)

    with open('voc.pkl', 'wb') as f:
        pickle.dump(voc, f)

    print("The files 'qa_pairs.pkl' and 'voc.pkl' are ready for training!")

if __name__ == "__main__":
    main()
