
# coding: utf-8

# In[431]:


import tensorflow as tf
import collections
from collections import namedtuple
import pickle
import parameters
import numpy as np


# In[432]:


#User2Vec = namedtuple('User2Vec', ['user_id', 'sent_ids', 'neg_ids', 'optimizer', 'loss', 'normalized_U'])
User2Vec = namedtuple('User2Vec', ['user_id', 'sent_ids', 'neg_ids', 'app', 'loss', 'normalized_U',"score"])


# In[433]:


def hinge_loss(user_embeds, word_embeds, neg_sample_ids):
    pos_score = tf.matmul(user_embeds, word_embeds, transpose_b = True)
    print('pos_score: ', pos_score)
    
    user_embeds_t = tf.transpose(user_embeds)
    neg_sample_ids_t = tf.transpose(neg_sample_ids)
    
    neg_score = tf.matmul(neg_sample_ids, user_embeds_t)
    #neg_score = tf.tensordot(neg_sample_ids_t, user_embeds_t, [0,0])
    print('neg_score: ', neg_score)

    loss = tf.maximum(0.0, 1 - tf.add(pos_score,neg_score))
    
    return loss


# In[434]:


def build_model(sess, graph, embed_matrix_rows, n_users, embed_matrix):
    lam = 1e-8
    with graph.as_default():
    # Ops and variables pinned to the CPU because of missing GPU implementation
        with tf.device('/cpu:0'):
            global_step = tf.Variable(0, trainable=False)
            
            # u_j
            user_id = tf.placeholder(tf.int32, shape=[1])
            print('user_ids: ', user_id)
            U = tf.Variable(tf.random_uniform([n_users, parameters.embedding_size], -1.0, 1.0))
            print('U: ', U)
            user_embed = tf.nn.embedding_lookup(U, user_id)
            #user_embed = tf.slice(U, [0, user_id], [U.get_shape()[0], 1])
#             user_embed = tf.transpose(user_embed)
            print('user_embed: ', user_embed)

            # e_i
            E = tf.Variable(embed_matrix, dtype=tf.float32)
            print('E: ', E)
            sent_ids = tf.placeholder(tf.int32, shape=None)
            print('sent_ids: ', sent_ids)
            word_embeds = tf.nn.embedding_lookup(E, sent_ids)
            print('word_embeds :', word_embeds)
            
            # e_l
            neg_ids = tf.placeholder(tf.int32, shape=None)
            neg_sample_ids = tf.nn.embedding_lookup(E, neg_ids)

            hinge_loss_1 =  hinge_loss(user_embed, word_embeds, neg_sample_ids)
            #U_regularizer = tf.nn.l2_loss(U)
            #E_regularizer = tf.nn.l2_loss(E)

            loss = tf.reduce_mean(hinge_loss_1 )#+ (lam/2) *  U_regularizer + (lam/2) *  E_regularizer)

            
        # Construct the SGD optimizer using a learning rate of 1.0.
        #optimizer = tf.train.GradientDescentOptimizer(1e-6).minimize(loss, global_step=global_step)
        
        
        #optimizer = tf.train.GradientDescentOptimizer(1e-6)
        #grads = optimizer.compute_gradients(loss)
        #clipped_grads = [(tf.clip_by_norm(grad, 5), var) for grad, var in grads]
        #app = optimizer.apply_gradients(clipped_grads)
        
        #AdamOptimizer
        
        #optimizer = tf.train.AdamOptimizer(1e-6)
        #grads = optimizer.compute_gradients(loss)
        #clipped_grads = [(tf.clip_by_norm(grad, 5), var) for grad, var in grads]
        #app = optimizer.apply_gradients(clipped_grads)
        
        #MomentumOptimizer
        optimizer = tf.train.MomentumOptimizer(1e-8,0.9)
        grads = optimizer.compute_gradients(loss)
        clipped_grads = [(tf.clip_by_norm(grad, 5), var) for grad, var in grads]
        app = optimizer.apply_gradients(clipped_grads)
        
        
        # Compute the cosine similarity between minibatch examples and all embeddings.
        norm = tf.sqrt(tf.reduce_sum(tf.square(U), 1, keep_dims=True))
        normalized_U = U / norm
        
        # generating score by adding the probabilities with which wrd2vec and user2vec were trained to do evaluation 
        # and finding the U with best score to stored for future use in CUE-CNN
        step_1 = tf.matmul(E,U,transpose_b = True)
        #print("step_1", step_1)
        step_2 = tf.nn.softmax(tf.transpose(step_1))
        step_3= tf.log(step_2)
        #print("step_2", step_2)
        score  = tf.reduce_mean(step_3,1)
        #print("score", score)
        
        tf.global_variables_initializer().run()
        print(" Train Initialized")

    #model = User2Vec(user_id, sent_ids, neg_ids, optimizer, loss, normalized_U)
    model = User2Vec(user_id, sent_ids, neg_ids, app, loss, normalized_U,score)
    
    return model


# In[435]:


def train(sess, model, n_users):
    
    user_ids = np.arange(n_users)
    max_num_steps = 10000
    #max_num_steps = 10
  
    user_idx = {}
    for prev_user, train, test, neg_samples in user_train_data:
        
        try:
            user_id = user_idx[prev_user]
        except KeyError:
            user_idx[prev_user] = len(user_idx)
        print('user: ', user_idx[prev_user])
        
        average_loss_step = max(parameters.checkpoint_step/10, 100)
    
        average_loss = 0
        
        for step in range(max_num_steps):
#             print('step: ', step)
              
            for id in np.random.permutation(len(train)):
                
#                 print('train[id]', len(train[id]))
                
#                 print('train[id]', len(neg_samples[id]))
#                 print('neg samples: ', neg_samples)
#                 print('train: ', train)
                
                for x in train[id]:
                    assert not np.any(np.isnan(x))
                #print('train: ', train[id])
                if train[id] == []:
                    continue
                feed_dict = {model.user_id.name: [user_idx[prev_user]], model.sent_ids.name: train[id],
                             model.neg_ids.name: neg_samples[id]}


            #_, loss_val = sess.run([model.optimizer, model.loss], feed_dict=feed_dict)
            _, loss_val = sess.run([model.app, model.loss], feed_dict=feed_dict)
            average_loss += loss_val
            
            if step % 1000 == 0:
                if step > 0:
                    average_loss /= 1000
                print("Average loss at step ", step, ": ", average_loss)
                average_loss = 0
            
                
                
    
    print("Train Finished")
    


# In[436]:


def evaluate(sess, model, n_users):
    user_ids = np.arange(n_users)
    max_num_steps = 100
    prev_logscore_n_users = float("-inf")
    #print("prev_logscore_n_users",type(prev_logscore_n_users))
    user_idx = {}
    for prev_user, train, test, neg_samples in user_train_data:
        
        try:
            user_id = user_idx[prev_user]
        except KeyError:
            user_idx[prev_user] = len(user_idx)
        #print("user",user_idx[prev_user])
        #print("\n\n")
        for step in range(max_num_steps):
            logscore_n_users = 0.0
            
            for msg in test:
                score = sess.run([model.score], feed_dict = {model.user_id.name: [user_idx[prev_user]], 
                                                             model.sent_ids.name: msg})
                score_uidx = score[0][user_idx[prev_user]]
                #print("Score_uidx",score_uidx)
                logscore_n_users += score_uidx
            print("avg logscore ", logscore_n_users/len(test))
            print("prev_logscore_n_users ", prev_logscore_n_users)
            if logscore_n_users/len(test) > prev_logscore_n_users:
                final_embeddings = model.normalized_U.eval()
                prev_logscore_n_users = logscore_n_users/len(test)
    print("Returning Best U")   
    return final_embeddings        
    
    


# In[437]:


if __name__ == '__main__':
    
    # load pickled word embeddings
    # because we want the number of users which we pickled here
    embed_matrix, unigram_prob, wrd2idx, word_counter, n_users = pickle.load(open(parameters.E_pkl, 'rb'))
       
    user_train_data = pickle.load(open(parameters.user_train_pkl, 'rb'))

    graph = tf.Graph()
    with tf.Session(graph=graph) as sess:

        model = build_model(sess, graph, embed_matrix.shape[0], n_users, embed_matrix)
        train(sess, model, n_users)
        user_embeddings = evaluate(sess,model, n_users)
    #print(user_embeddings)
    pickle.dump(user_embeddings, open('user_embeddings.pkl', 'wb'))

