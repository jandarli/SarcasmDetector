
# coding: utf-8

# In[33]:


#import build_train 
import pickle
import numpy as np
import tensorflow as tf
import parameters
from collections import defaultdict
from nltk.tokenize import TweetTokenizer
from nltk import tokenize
import re
import math
max_len = 20
batch_size = 10
hidden_size = 25
embedding_size = 128
#pooled_outputs = []


# In[46]:


def create_sent_corpus():
    line_split = []
    message_list = defaultdict(list)
    with open("cleaned_data.txt","r") as fid:
            c = 0
            for line in fid:
                c+=1
                #if c > 100:
                #    break
    #             print("len", len(line))
                if len(line) > 1:
                    line_split = line.split(" ")
                    message_line = " ".join(line_split[2:])
                    
                    #message_list.append(" ".join(line.split()[2:]))
                    message_list[line_split[0]].append((message_line,line_split[1]))
            #max_len = len(max(message_list, key=len))
            #print(message_list)
            print(" total ",c," lines read fron train_labels")
            return  message_list


# In[36]:


def genTrainExamples(message_list, max_len, wrd2idx):
    #feature_list = defaultdict(list)
    #found max_len greater that embedding size don't understand a thing about it. Keeping it 10 temporarily.
    feature_list = []
    c = 0
    sent_list = []
    tokenizer = TweetTokenizer(preserve_case=False)
    regex = re.compile(r'[\.\]\%\[\'",\?\*!\}\{<>\^-]')
    for key, message_line_list in message_list.items():
        #c += 1
        #if c > 70:
        #    break
        for sent,label in message_line_list:
            content = tokenizer.tokenize(sent)
            content = [word for word in content if not regex.match(word)]
            for w in content:                 
                if w in wrd2idx: #don't know if we have to take words comming only in wrd2idx(topwords)
                    sent_list.append(wrd2idx[w])
            while len(sent_list) < max_len:
                sent_list.append(0)
            while len(sent_list) > max_len:
                sent_list.pop()
            assert len(sent_list) == max_len
            feature_list.append((key,sent_list,label))
            sent_list = []  
    #print(feature_list)
    return feature_list
                


# Try dense layer, try to figure out why reshape has to use 3*3, try to figure out drop out and also try to understand what happens if we give same heights to all filters, try to figure out the strides, also for varying heights try different height combinations. 
# 
# Used this remember to put this in refference if the structure eremains the same for cnn in future.
# http://www.wildml.com/2015/12/implementing-a-cnn-for-text-classification-in-tensorflow/

# In[47]:


class CNNModel(object):
     
    
    def __init__(self, graph, E, U):

        
        self.build_graph(graph, E, U)

    def build_graph(self, graph, E, U):
        """

        :param graph:
        :param embedding_array:
        :param Config:
        :return:
        """
        pooled_outputs = []
        with graph.as_default():
            # for CNN #####
            #pooled_outputs = []
            self.embeddings = tf.Variable(E, dtype=tf.float32)
            self.train_inputs = tf.placeholder(tf.int32, shape=[None,max_len])
            print("train_inputs",self.train_inputs )
            self.train_labels = tf.placeholder(tf.int32, shape=[None,1])
            print("train_labels",self.train_labels )
            embed = tf.nn.embedding_lookup(self.embeddings, self.train_inputs)
            print("embed after lookup",embed)
            embed = tf.expand_dims(embed, -1)
            total_h_grams = 3 * 100
            
            
            # for RNN ###
            self.user_embeddings = tf.Variable(U, dtype=tf.float32)
            self.user_id = tf.placeholder(tf.int32, shape=[batch_size,])
            print("user_id",self.user_id )
            user_embed = tf.nn.embedding_lookup(self.user_embeddings, self.user_id)
            print("user_embed after lookup",user_embed)
            
            hidden_layer_weights = tf.Variable(tf.truncated_normal([hidden_size, total_h_grams + embedding_size],
                                                                      stddev=1.0 / math.sqrt(hidden_size)))
            print("hidden layer weights",hidden_layer_weights)
            hidden_layer_bias = tf.Variable(tf.zeros([hidden_size,1]))
            
            print("hidden_layer_bias",hidden_layer_bias )
            ####################### was 2 in paper but have taken 1######################
            output_layer_weights = tf.Variable(tf.random_normal([1 ,hidden_size ],
            
                                                                   stddev=1.0 / math.sqrt(hidden_size)))
            print("output_layer_weights",output_layer_weights)  
            output_layer_bias = tf.Variable(tf.zeros([1,1]))
            print("output_layer_bias",output_layer_bias)         
            
            # CNN #####
            for f_size in [1,3,5]:                
                filter_weights = tf.Variable(tf.truncated_normal([f_size, 128, 1, 100], stddev=0.1))
                bias_conv = tf.Variable(tf.zeros(100))
                conv = tf.nn.conv2d(embed,filter_weights,strides=[1, 1, 1, 1],padding="VALID")
                additive_bias = tf.nn.bias_add(conv, bias_conv)
                Relu_layer = tf.nn.relu(additive_bias)
                pooled = tf.nn.max_pool(Relu_layer,ksize=[1, max_len - f_size + 1, 1, 1],strides=[1, 1, 1, 1],padding='VALID')
                pooled_outputs.append(pooled)
            
            print(len(pooled_outputs))
            
            concatenate_map_filters = tf.concat(pooled_outputs,3)
            pooled_outputs = []
            print("Concatenated map filters", concatenate_map_filters)
            Cs_matrix = tf.reshape(concatenate_map_filters,[total_h_grams, -1])
            print("Cs_matrx",Cs_matrix)
            
            # RNN ###
            reshaped_user_embed = tf.reshape(user_embed,[embedding_size,-1])
            print("reshaped_user_embed", reshaped_user_embed)
            rnn_input = tf.concat([reshaped_user_embed,Cs_matrix],0)
            print("rnn_input",rnn_input)
            H_Cs_U = tf.matmul(hidden_layer_weights,rnn_input)
            print("H_Cs_U",H_Cs_U)
            hidden_layer_output = tf.add(H_Cs_U,hidden_layer_bias)
            print("hidden_layer_output",hidden_layer_output)
            activation_layer_output = tf.nn.relu(hidden_layer_output)
            
            print("activation_layer_output",activation_layer_output)
            self.prediction = tf.add(tf.matmul(output_layer_weights,activation_layer_output),output_layer_bias)
            print("prediction",self.prediction)
            self.loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=tf.transpose(self.prediction),
                                                                                         labels=self.train_labels))
            """
            cue_cnn_loss = tf.nn.softmax_cross_entropy_with_logits_v2(logits=tf.transpose(self.prediction),
                                                                                          labels=self.train_labels)
            
            
            filter_weight_regularizer = tf.nn.l2_loss(filter_weights)
            
            h_weight_regularizer = tf.nn.l2_loss(hidden_layer_weights)
            h_bias_regularizer = tf.nn.l2_loss(hidden_layer_bias)
            
            output_layer_weights_regularizer = tf.nn.l2_loss(output_layer_weights)
            output_layer_bias_regularizer = tf.nn.l2_loss(output_layer_bias)
            

            self.loss = tf.reduce_mean(cue_cnn_loss + (1e-6/2) *  filter_weight_regularizer 
                                                    + (1e-6/2) *  h_weight_regularizer
                                                    + (1e-6/2) *  h_bias_regularizer
                                                    + (1e-6/2) *  output_layer_weights_regularizer
                                                    + (1e-6/2) *  output_layer_bias_regularizer
                                       
                                                    )
            
            """
            optimizer = tf.train.GradientDescentOptimizer(1e-6)
            grads = optimizer.compute_gradients(self.loss)
            clipped_grads = [(tf.clip_by_norm(grad, 5), var) for grad, var in grads]
            self.app = optimizer.apply_gradients(clipped_grads)
            
            
            self.init = tf.global_variables_initializer()
            
  
            
    def train(self, sess, trainFeats, max_len ):
        self.init.run()
        print("Initailized")
        c = 0
        max_num_steps = 1000
        user_idx = {}  
        user  = []
        sent  = []
        label = []
        for tuple_i in trainFeats:
            try:
                u_id = user_idx[tuple_i[0]]
            except KeyError:
                user_idx[tuple_i[0]] = len(user_idx)
            u_id = user_idx[tuple_i[0]]
            if u_id > 84: ################################### ADDED because of the error in training of user2vec
                break
            user.append(u_id)
            
            sent_i , label_i = tuple_i[1], tuple_i[2]
            sent.append(sent_i)
            label.append([float(label_i)])
        average_loss = 0
        print("label",len(label))
        print("sent",len(sent))
        print("user",len(user))
        print("trainFeats",len(trainFeats))
        for step in range(max_num_steps):
            start = (step * batch_size) % len(user)
            end = ((step + 1) * batch_size) % len(user)
            if end < start:
                start -= end
                end = len(user)
            batch_user, batch_inputs, batch_labels = user[start:end], sent[start:end], label[start:end]
            len(batch_inputs)
            #feed_dict = {self.train_inputs: batch_inputs, self.train_labels: batch_labels, self.prob: 0.5}
            feed_dict = {self.user_id.name:batch_user, self.train_inputs.name: batch_inputs,self.train_labels.name: batch_labels}
            _, loss_val = sess.run([self.app, self.loss], feed_dict=feed_dict)
            average_loss += loss_val
            #print("loss_val",loss_val)
            if step % 100 == 0:
                    if step > 0:
                        average_loss /= 100
                    print("Average loss at step ", step, ": ", average_loss)
                    average_loss = 0

        print("Train Finished.")

            
        """
            for step in range(max_num_steps):
                
                
                
                #k=np.asarray(sent)
                #sent_t = np.transpose(k)
                print("sent",sent)
                print("label",label)
                #for i in range(len(label)):
                    #print("label",label[i])
                    #print("sent",sent[i])
                feed_dict = {self.user_id.name:[u_id], self.train_inputs.name: sent,self.train_labels.name: label}
                #pooled_outputs = []
                _, loss_val = sess.run([self.app, self.cue_cnn_loss], feed_dict=feed_dict)
                #print("type of loss val",loss_val)
                average_loss += loss_val[0]

                if step % 1 == 0:
                    if step > 0:
                        average_loss /= 1
                    print("Average loss at step ", step, ": ", average_loss)
                    average_loss = 0
            #print("next")
            c +=1
        print("train finished")
        #print(c)
        #print(len(self.pooled_outputs))
        #return self.pooled_outputs
           
        """


# In[48]:


def init_cnn():
    E,unigram_prob,wrd2idx,word_counter,n_users = pickle.load(open('train_embeddings.pkl', 'rb'))
    U = pickle.load(open("user_embeddings.pkl","rb"))
    print(E.shape)
    message_list = create_sent_corpus()
    print("Generating Traning Examples")
    trainFeats= genTrainExamples(message_list, max_len, wrd2idx)
    #print(trainFeats)
    print("Done.")

    # Build the graph model
    graph = tf.Graph()

    model = CNNModel(graph, E, U)

    with tf.Session(graph=graph) as sess:
        
        Cs_matrix = model.train(sess, trainFeats, max_len)
        
    


# In[50]:


init_cnn()

