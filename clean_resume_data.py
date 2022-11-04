# Import the libraries. #
import os
import unidecode
import pickle as pkl

from collections import Counter
from nltk import wordpunct_tokenize

# Load the data. #
tmp_resume_folder = \
    "../../Data/resume/resumes_corpus/"
tmp_resume_files  = [x for x in os.listdir(
    tmp_resume_folder) if x.endswith(".txt")]
print(len(tmp_resume_files), "resumes found.")

# Extract the vocabulary. #
min_length = 10
w_counter  = Counter()
resume_text = []
for tmp_resume_file in tmp_resume_files:
    tmp_open_file  = tmp_resume_folder + tmp_resume_file
    with open(tmp_open_file, "rb") as tmp_load:
        tmp_resume_txt = unidecode.unidecode(
            tmp_load.read().decode("utf-8", errors="ignore"))
    
    tmp_tokens = [
        x for x in wordpunct_tokenize(
            tmp_resume_txt.lower().strip()) if x != ""]
    tmp_tokens += ["[EOS]"]
    
    if len(tmp_tokens) >= min_length:
        w_counter.update(tmp_tokens)
        resume_text.append(" ".join(tmp_tokens))

# Only use words which occur more than 5 times. #
min_count  = 5
word_vocab = ["[SOS]", "[EOS]", "[UNK]", "[PAD]"]
word_vocab += list(sorted([
    x for x, y in w_counter.most_common() if y > min_count]))

word_2_idx = dict([
    (word_vocab[x], x) for x in range(len(word_vocab))])
idx_2_word = dict([
    (x, word_vocab[x]) for x in range(len(word_vocab))])
print("Vocabulary Size:", len(word_vocab), "words.")

# Save the data. #
resume_data = {
    "word_vocab": word_vocab, 
    "word_2_idx": word_2_idx, 
    "idx_2_word": idx_2_word, 
    "resume_text": resume_text
}

tmp_pkl_file = tmp_resume_folder
tmp_pkl_file += "resume_data.pkl"
with open(tmp_pkl_file, "wb") as tmp_save:
    pkl.dump(resume_data, tmp_save)

print("Data saved to", tmp_pkl_file + ".")