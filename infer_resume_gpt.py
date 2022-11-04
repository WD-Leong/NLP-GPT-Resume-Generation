import os
import time
import numpy as np
import pickle as pkl
from nltk import wordpunct_tokenize

import tensorflow as tf
import tensorflow_addons as tfa
import tf_ver2_gpt_keras_v2 as tf_gpt

# Force to use CPU. #
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = ""

# Model Parameters. #
prob_keep  = 0.9
num_heads  = 4
num_layers = 3
seq_length = 100

hidden_size = 256
ffwd_size   = 4*hidden_size

model_ckpt_dir  = "TF_Models/resume_gpt"
train_loss_file = "train_loss_resume_gpt.csv"

# Load the data. #
tmp_pkl_file = \
    "../../Data/resume/resumes_corpus/"
tmp_pkl_file += "resume_data.pkl"
with open(tmp_pkl_file, "rb") as tmp_load:
    resume_data = pkl.load(tmp_load)

# Extract the data and its assets. #
word_vocab = resume_data["word_vocab"]
word_2_idx = resume_data["word_2_idx"]
idx_2_word = resume_data["idx_2_word"]
vocab_size = len(word_vocab)

# Define the special tokens. #
SOS_token = word_2_idx["[SOS]"]
EOS_token = word_2_idx["[EOS]"]
PAD_token = word_2_idx["[PAD]"]
UNK_token = word_2_idx["[UNK]"]

# Set the number of threads to use. #
tf.config.threading.set_intra_op_parallelism_threads(1)
tf.config.threading.set_inter_op_parallelism_threads(1)

# Build the GPT model. #
print("Loading the GPT Keras Model.")
start_time = time.time()

gpt_model = tf_gpt.GPTDecoder(
    num_layers, num_heads, 
    hidden_size, ffwd_size, vocab_size, 
    seq_length, rate1=0.0, rate2=1.0-prob_keep)
gpt_optimizer = tfa.optimizers.AdamW(
    weight_decay=1.0e-4)

# Create the model checkpoint. #
ckpt = tf.train.Checkpoint(
    step=tf.Variable(0), 
    gpt_model=gpt_model, 
    gpt_optimizer=gpt_optimizer)

manager = tf.train.CheckpointManager(
    ckpt, model_ckpt_dir, max_to_keep=1)

ckpt.restore(manager.latest_checkpoint)
if manager.latest_checkpoint:
    print("Model restored from {}".format(
        manager.latest_checkpoint))
else:
    print("Error: No latest checkpoint found.")

tmp_init = gpt_model(np.zeros(
    [1, seq_length], dtype=np.int32), training=False)
elapsed_time = (time.time()-start_time) / 60
del tmp_init

print(gpt_model.summary())
print("GPT Model Built", 
      "(" + str(elapsed_time) + " mins).")
print("-" * 75)

n_iter = ckpt.step.numpy().astype(np.int32)
print("Model Inference", "(" + str(n_iter), "iterations).")
print("-" * 75)

while True:
    tmp_prompt = input("Enter prompt: ")
    tmp_prompt = str(tmp_prompt).lower().strip()
    
    if tmp_prompt == "":
        break
    else:
        gen_seed = [
            word_2_idx.get(x, UNK_token) for \
                x in wordpunct_tokenize(tmp_prompt)]
        seed_array = np.array(gen_seed, dtype=np.int32)
        
        infer_ids = gpt_model.gen_text(
            seed_array.reshape((1, -1)), 
            250, sample=False).numpy()[0]
        
        infer_toks = [idx_2_word[x] for x in infer_ids]
        infer_text = " ".join(
            infer_toks).replace("[PAD]", "").strip()
        print("Generated Text:", infer_text)
        print("-" * 75)
