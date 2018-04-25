
# coding: utf-8

# In[1]:


embedding_size = 128
max_num_steps  = 200001
    
checkpoint_step = 50000
vocabulary_size = 100000
learning_rate = 0.01

checkpoint_step = 50000

embed = "word2vec_nce.model"
train_file = "user_text_sample.txt"
output_pkl = "train_embeddings.pkl"
vocabulary_size = 100000
min_word_freq = 5
seed = 42
neg_samples = 10
output = "user_train_data.pkl"

# number of words from previous user to put in training
sent_idx = 15
# neg_idx = 

