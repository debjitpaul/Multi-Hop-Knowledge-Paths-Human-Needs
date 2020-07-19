__author__ = 'debjit'

import collections
import tensorflow as tf
import numpy
import re
import os
from tensorflow.python.framework import ops
from tensorflow.python.ops import math_ops
import random
import ast
import numpy as np
import keras
from sklearn.metrics import hamming_loss
import math
from numpy import array
import tensorflow_hub as hub
import codecs
import nltk
from nltk.tokenize import TweetTokenizer
from nltk.stem import WordNetLemmatizer 
lemmatizer = WordNetLemmatizer() 
tknzr = TweetTokenizer()
try:
    import cPickle as pickle
except:
    import pickle

class Human_needs(object):
    def __init__(self, config):
        self.config = config
        self.UNK = "<unk>"
        self.word2id = None
        self.embedding_matrix=[]
        self.term2index = None
        self.index2term = None

    def build_vocabs(self, data_train, data_dev, data_test, dim_embedding, embedding_path=None):
    
        data_source = list(data_train)    
        if self.config["vocab_include_devtest"]:
            if data_dev != None:
                data_source += data_dev
            if data_test != None:
                data_source += data_test
        
        id, sentences, lst2, weight_per, context, label_distribution= zip(*data_source) 
        
        wp_vocab = set(token for sent in sentences for token in sent.split(' '))
        wp_vocab_knw = set(token for s in lst2 for sent in s for token in sent.split(' '))
        wp_vocab_context = set(token for sent in context for token in sent.split(' '))
    
        wp_vocab = wp_vocab.union(wp_vocab_knw)
        wp_vocab = wp_vocab.union(wp_vocab_context)
        #dim_embedding = 100
        unk = numpy.random.uniform(-0.2, 0.2, dim_embedding)
        embeddings = {'UNK': unk}
        embed_file= codecs.open(embedding_path,encoding='utf-8',mode='r')
        lines = embed_file.readlines()
        pre_embed={}
        self.term2index={}
        count=0
        
        word_counter = collections.Counter()
        
        for word in sentences:
          for token in word.split(' '):
                w = token
                if self.config["lowercase"] == True:
                    w = w.lower()
                if self.config["replace_digits"] == True:
                    w = re.sub(r'\d', '0', w)
                word_counter[w] += 1
                self.term2index[w] = count
                count = count + 1
                
        for word in context:
            for token in word.split(' '):
                w = token
                if self.config["lowercase"] == True:
                    w = w.lower()
                if self.config["replace_digits"] == True:
                    w = re.sub(r'\d', '0', w)
                word_counter[w] += 1
                self.term2index[w] = count
                count = count + 1
                
        for para in lst2:        
         for sent in para:
           for token in word.split(' '):
                w = token
                if self.config["lowercase"] == True:
                    w = w.lower()
                if self.config["replace_digits"] == True:
                    w = re.sub(r'\d', '0', w)
                word_counter[w] += 1
                self.term2index[w] = count
                count = count + 1

        self.word2id = collections.OrderedDict([(self.UNK, 0)])
        for word, count in word_counter.most_common():
            if self.config["min_word_freq"] <= 0 or count >= self.config["min_word_freq"]:
                if word not in self.word2id:
                    self.word2id[word] = len(self.word2id)
        
        if embedding_path != None and self.config["vocab_only_embedded"] == True:
            self.embedding_vocab = set([self.UNK])
            with codecs.open(embedding_path, encoding='utf-8',mode='r') as f:
                for line in f:
                    line_parts = line.strip().split()
                    if len(line_parts) <= 2:
                        continue
                    w = line_parts[0]
                    if self.config["lowercase"] == True:
                        w = w.lower()
                    if self.config["replace_digits"] == True:
                        w = re.sub(r'\d', '0', w)
                    self.embedding_vocab.add(w)
            
            word2id_revised = collections.OrderedDict()
            for word in self.word2id:
                if word in embedding_vocab and word not in word2id_revised:
                    word2id_revised[word] = len(word2id_revised)
            self.word2id = word2id_revised
        
        self.index2term={}
        self.term2index = self.word2id
        self.index2term = {v:k for k,v in self.term2index.items()}
        print("n_words: " + str(len(list(wp_vocab)))) 
        
        
    def construct_network(self):
    
    
        tf.reset_default_graph()
        self.word_ids = tf.placeholder(tf.int32, [None, None], name="word_ids")
        self.sentence_lengths = tf.placeholder(tf.int32, [None], name="sentence_lengths")
        self.word_ids_knowledge = tf.placeholder(tf.int32, [None, None, None], name="word_ids_know")
        self.sentence_tokens = tf.placeholder(tf.string, [None, None], name="word_list_sentence")
        self.knowledge_lengths = tf.placeholder(tf.int32, [None, None], name="sentence_lengths_know")
        self.knowledge_tokens = tf.placeholder(tf.string, [None, None, None], name="word_list_knowledge")
        self.knowledge_max_lengths = tf.placeholder(tf.int32, [None, None], name="sentence_lengths_max_know")
        self.word_ids_context = tf.placeholder(tf.int32, [None, None], name="word_ids_context")
        self.context_tokens = tf.placeholder(tf.string, [None, None], name="words_list_context")
        self.context_lengths = tf.placeholder(tf.int32, [None], name="sentence_lengths_context")
        self.sentence_labels = tf.placeholder(tf.float32, [None, None], name="sentence_labels")
        self.batch_size = tf.Variable(0)
        self.max_lengths = tf.placeholder(tf.int32, [None], name="max_lengths_padding")
        self.weights_path = tf.placeholder(tf.float32, [None, None], name="weights_path")
        self.learningrate = tf.placeholder(tf.float32, name="learningrate")
        self.is_training = tf.placeholder(tf.bool, name="is_training")
        self.loss = 0.0
        input_tensor = None
        input_vector_size = 0 
        #reiss= ['physiological', 'love', 'spiritual growth', 'esteem', 'stability']
        if self.config["human_needs"] == "maslow":
            reiss=['physiological', 'love', 'spiritual growth', 'esteem', 'stability']
            #human_needs =['physiological', 'love', 'spiritual growth', 'esteem', 'stability']
        elif self.config["human_needs"] == "reiss":
            #reiss = ['status', 'approval', 'tranquility', 'competition', 'health', 'family', 'romance', 'food', 'indep', 'power', 'order', 'curiosity', 'serenity', 'honor', 'belonging', 'contact', 'savings', 'idealism', 'rest']
            reiss = ['status', 'approval', 'tranquility', 'competition', 'health', 'family', 'romance', 'food', 'indep', 'power', 'order', 'curiosity', 'serenity', 'honor', 'contact', 'savings', 'idealism', 'rest']
                
        self.initializer = None
        if self.config["initializer"] == "normal":
            self.initializer = tf.random_normal_initializer(mean=0.0, stddev=0.1)
        elif self.config["initializer"] == "glorot":
            self.initializer = tf.glorot_uniform_initializer()
        elif self.config["initializer"] == "xavier":
            self.initializer = tf.glorot_normal_initializer()
            
            
            
            
###############################################################################    BILSTM   #############################################################################################
        if self.config["neural_network"]=="BILSTM":
###############################################################################   SENTENCE BI-LSTM  #############################################################################################
         zeros_initializer = tf.zeros_initializer()
         input_tensor = None
         with tf.variable_scope("sentence"):
          word_lstm_cell_fw = tf.nn.rnn_cell.LSTMCell(self.config["word_recurrent_size"], 
            use_peepholes=self.config["lstm_use_peepholes"], 
            state_is_tuple=True, 
            initializer=self.initializer,
            reuse=False)
            
          word_lstm_cell_bw = tf.nn.rnn_cell.LSTMCell(self.config["word_recurrent_size"], 
            use_peepholes=self.config["lstm_use_peepholes"], 
            state_is_tuple=True, 
            initializer=self.initializer,
            reuse=False)
            
          self.word_embeddings = tf.get_variable("word_embeddings", 
                                 shape=[len(self.term2index), self.config["word_embedding_size"]], 
                                 initializer=(zeros_initializer if self.config["emb_initial_zero"] == True else self.initializer), 
                                 trainable=(True if self.config["train_embeddings"] == True else False))
          use_elmo = True
          if use_elmo:
          	elmo = hub.Module("https://tfhub.dev/google/elmo/2", trainable=True)   
          	input_tensor = elmo(inputs={"tokens": self.sentence_tokens,"sequence_len": self.sentence_lengths},signature="tokens",as_dict=True)["elmo"]
          else:
          	input_tensor = tf.nn.embedding_lookup(self.word_embeddings, self.word_ids)          
          
          input_vector_size = self.config["word_embedding_size"]
          self.word_representations = input_tensor
          dropout_input = self.config["dropout_input"] * tf.cast(self.is_training, tf.float32) + (1.0 - tf.cast(self.is_training, tf.float32))
          input_tensor =  tf.nn.dropout(input_tensor, dropout_input, name="dropout_word")
                
          (lstm_outputs_fw, lstm_outputs_bw), ((_, lstm_output_fw), (_, lstm_output_bw)) = tf.nn.bidirectional_dynamic_rnn(word_lstm_cell_fw, word_lstm_cell_bw, input_tensor, sequence_length=self.sentence_lengths, dtype=tf.float32, time_major=False)
          
          dropout_word_lstm = self.config["dropout_word_lstm"] * tf.cast(self.is_training, tf.float32) + (1.0 - tf.cast(self.is_training, tf.float32))
          lstm_outputs_fw =  tf.nn.dropout(lstm_outputs_fw, dropout_word_lstm, noise_shape=tf.convert_to_tensor([tf.shape(self.word_ids)[0],1,self.config["word_recurrent_size"]], dtype=tf.int32))
          lstm_outputs_bw =  tf.nn.dropout(lstm_outputs_bw, dropout_word_lstm, noise_shape=tf.convert_to_tensor([tf.shape(self.word_ids)[0],1,self.config["word_recurrent_size"]], dtype=tf.int32))
          lstm_outputs = tf.concat([lstm_outputs_fw, lstm_outputs_bw], -1)
          self.lstm_outputs = lstm_outputs
          
          if self.config["sentence_composition"] == "last":
                processed_tensor = lstm_outputs
                self.attention_weights_unnormalised = tf.zeros_like(self.word_ids, dtype=tf.float32)
          elif self.config["sentence_composition"] == "attention":
                attention_evidence = tf.layers.dense(lstm_outputs, self.config["attention_evidence_size"], activation=tf.sigmoid, kernel_initializer=self.initializer)
                attention_weights = tf.layers.dense(attention_evidence, 1, activation=None, kernel_initializer=self.initializer)
                attention_weights = tf.reshape(attention_weights, shape=tf.shape(self.word_ids))
                if self.config["attention_activation"] == "sharp":
                    attention_weights = tf.exp(attention_weights)
                elif self.config["attention_activation"] == "soft":
                    attention_weights = tf.sigmoid(attention_weights)
                elif self.config["attention_activation"] == "linear":
                    pass
                else:
                    raise ValueError("Unknown activation for attention: " + str(self.config["attention_activation"]))

                self.attention_weights_unnormalised = attention_weights
                attention_weights = tf.where(tf.sequence_mask(self.sentence_lengths), attention_weights, tf.zeros_like(attention_weights))
                attention_weights = attention_weights / tf.reduce_sum(attention_weights, 1, keep_dims=True)
                processed_tensor_1 = tf.reduce_sum(lstm_outputs * attention_weights[:,:,numpy.newaxis], 1)

          
          self.token_scores = [tf.where(tf.sequence_mask(self.sentence_lengths), self.attention_weights_unnormalised, tf.zeros_like(self.attention_weights_unnormalised) - 1e6)]
          
          if self.config["hidden_layer_size"] > 0:
             if self.config["sentence_composition"] == "attention":
                #processed_tensor_sentence = tf.reduce_max(lstm_outputs,1) 
                processed_tensor_sentence = tf.layers.dense(processed_tensor_1, self.config["hidden_layer_size"], activation=tf.nn.relu, kernel_initializer=self.initializer)
             elif self.config["sentence_composition"] == "last": 
               processed_tensor_sentence = tf.layers.dense(processed_tensor, self.config["hidden_layer_size"], activation=tf.nn.relu, kernel_initializer=self.initializer)
          
          
#####################################################################  CONTEXT BI-LSTM ##################################################   
          
         with tf.variable_scope("context"):      
          
          context_lstm_cell_fw = tf.nn.rnn_cell.LSTMCell(self.config["word_recurrent_size"], 
            use_peepholes=self.config["lstm_use_peepholes"], 
            state_is_tuple=True, 
            initializer=self.initializer,
            reuse=False)
            
          context_lstm_cell_bw = tf.nn.rnn_cell.LSTMCell(self.config["word_recurrent_size"], 
            use_peepholes=self.config["lstm_use_peepholes"], 
            state_is_tuple=True, 
            initializer=self.initializer,
            reuse=False)    
          input_vector_size = self.config["word_embedding_size"]
          self.word_representations = input_tensor
          
          use_elmo = True
          if use_elmo:
                elmo = hub.Module("https://tfhub.dev/google/elmo/2", trainable=True)     
                input_tensor= elmo(inputs={"tokens": self.context_tokens,"sequence_len": self.context_lengths},signature="tokens",as_dict=True)["elmo"]
          else:
          	input_tensor = tf.nn.embedding_lookup(self.word_embeddings, self.word_ids_context)  
          dropout_input = self.config["dropout_input"] * tf.cast(self.is_training, tf.float32) + (1.0 - tf.cast(self.is_training, tf.float32))
          input_tensor =  tf.nn.dropout(input_tensor, dropout_input, name="dropout_word")	    
          (lstm_outputs_fw, lstm_outputs_bw), ((_, lstm_output_fw), (_, lstm_output_bw)) = tf.nn.bidirectional_dynamic_rnn(context_lstm_cell_fw, context_lstm_cell_bw, input_tensor, sequence_length=self.context_lengths, dtype=tf.float32, time_major=False)
          
          dropout_word_lstm = self.config["dropout_word_lstm"] * tf.cast(self.is_training, tf.float32) + (1.0 - tf.cast(self.is_training, tf.float32))
          lstm_outputs_fw =  tf.nn.dropout(lstm_outputs_fw, dropout_word_lstm, noise_shape=tf.convert_to_tensor([tf.shape(self.word_ids_context)[0],1,self.config["word_recurrent_size"]], dtype=tf.int32))
          lstm_outputs_bw =  tf.nn.dropout(lstm_outputs_bw, dropout_word_lstm, noise_shape=tf.convert_to_tensor([tf.shape(self.word_ids_context)[0],1,self.config["word_recurrent_size"]], dtype=tf.int32))
          lstm_outputs = tf.concat([lstm_outputs_fw, lstm_outputs_bw], -1)
          #if self.config["hidden_layer_size"] > 0:
          #      lstm_outputs = tf.layers.dense(lstm_outputs, self.config["hidden_layer_size"], activation=tf.nn.relu, kernel_initializer=self.initializer)
          self.lstm_outputs = lstm_outputs

          if self.config["sentence_composition"] == "last":
                processed_tensor_context = lstm_outputs
                self.attention_weights_unnormalised = tf.zeros_like(self.word_ids_context, dtype=tf.float32)
          elif self.config["sentence_composition"] == "attention":      
                attention_evidence = tf.layers.dense(lstm_outputs, self.config["attention_evidence_size"], activation=tf.sigmoid, kernel_initializer=self.initializer)

                attention_weights = tf.layers.dense(attention_evidence, 1, activation=None, kernel_initializer=self.initializer)
                attention_weights = tf.reshape(attention_weights, shape=tf.shape(self.word_ids_context))

                if self.config["attention_activation"] == "sharp":
                    attention_weights = tf.softmax(attention_weights)
                elif self.config["attention_activation"] == "soft":
                    attention_weights = tf.sigmoid(attention_weights)
                elif self.config["attention_activation"] == "linear":
                    pass
                else:
                    raise ValueError("Unknown activation for attention: " + str(self.config["attention_activation"]))

                self.attention_weights_unnormalised = attention_weights
                attention_weights = tf.where(tf.sequence_mask(self.context_lengths), attention_weights, tf.zeros_like(attention_weights))
                attention_weights = attention_weights / tf.reduce_sum(attention_weights, 1, keep_dims=True)
                processed_tensor_context = tf.reduce_sum(lstm_outputs * attention_weights[:,:,numpy.newaxis], 1)
          if self.config["hidden_layer_size"] > 0:
              #processed_tensor_context = tf.reduce_mean(lstm_outputs,1)
              processed_tensor_context = tf.layers.dense(processed_tensor_context, self.config["hidden_layer_size"], activation=tf.nn.relu, kernel_initializer=self.initializer)


####################################################################### KNOWLEDGE Bi-LSTM ####################################################################################################
         processed_tensor_1 = processed_tensor_sentence
         with tf.variable_scope("knowledge"):
           knowledge_input_tensor = tf.nn.embedding_lookup(self.word_embeddings, self.word_ids_knowledge)
           input_vector_size = self.config["word_embedding_size"]
           
           know_lstm_cell_fw = tf.nn.rnn_cell.LSTMCell(self.config["word_embedding_size"],
                    use_peepholes=self.config["lstm_use_peepholes"], 
                    state_is_tuple=True, 
                    initializer=self.initializer,
                    reuse=False)
           know_lstm_cell_bw = tf.nn.rnn_cell.LSTMCell(self.config["word_embedding_size"],
                    use_peepholes=self.config["lstm_use_peepholes"], 
                    state_is_tuple=True, 
                    initializer=self.initializer,
                    reuse=False)
           
           self.word_representations = knowledge_input_tensor
           s = tf.shape(knowledge_input_tensor)
           knowledge_input_tensor = tf.reshape(knowledge_input_tensor, shape=[s[0]*s[1], s[2], self.config["word_embedding_size"]])
           knowledge_lengths = tf.reshape(self.knowledge_max_lengths, shape=[s[0]*s[1]])
           dropout_input = self.config["dropout_input"] * tf.cast(self.is_training, tf.float32) + (1.0 - tf.cast(self.is_training, tf.float32))
           knowledge_input_tensor =  tf.nn.dropout(knowledge_input_tensor, dropout_input, name="dropout_word")
              
           char_lstm_outputs = tf.nn.bidirectional_dynamic_rnn(know_lstm_cell_fw, know_lstm_cell_bw, knowledge_input_tensor, sequence_length=knowledge_lengths, dtype=tf.float32, time_major=False)
           _, ((_, char_output_fw), (_, char_output_bw)) = char_lstm_outputs
           lstm_outputs = tf.concat([char_output_fw, char_output_bw], -1)
          '''                        
           if self.config["sentence_composition"] == "attention":      
                attention_evidence = tf.layers.dense(lstm_outputs, self.config["attention_evidence_size"], activation=tf.sigmoid, kernel_initializer=self.initializer)
                attention_weights = tf.layers.dense(attention_evidence, 1, activation=None, kernel_initializer=self.initializer)
                attention_weights = tf.reshape(attention_weights, shape=tf.shape(self.word_ids_knowledge))
                if self.config["attention_activation"] == "sharp":
                    attention_weights = tf.softmax(attention_weights)
                elif self.config["attention_activation"] == "soft":
                    attention_weights = tf.sigmoid(attention_weights)
                elif self.config["attention_activation"] == "linear":
                    pass
                else:
                    raise ValueError("Unknown activation for attention: " + str(self.config["attention_activation"]))

                self.attention_weights_unnormalised = attention_weights
                attention_weights = tf.where(tf.sequence_mask(self.knowledge_max_lengths), attention_weights, tf.zeros_like(attention_weights))
                attention_weights = attention_weights / tf.reduce_sum(attention_weights, 1, keep_dims=True)
                atten_shape = tf.shape(attention_weights)
                attention_weights = tf.reshape(attention_weights, shape=[tf.shape(attention_weights)[0]*tf.shape(attention_weights)[1],tf.shape(attention_weights)[2]])
                lstm_outputs = tf.reduce_sum(lstm_outputs * attention_weights[:,:,numpy.newaxis], 1)
          
           '''
           lstm_outputs = tf.reshape(lstm_outputs, shape=[s[0], s[1], 2*self.config["word_embedding_size"]])
           dropout_word_lstm = self.config["dropout_word_lstm"] * tf.cast(self.is_training, tf.float32) + (1.0 - tf.cast(self.is_training, tf.float32))
           lstm_outputs = tf.nn.dropout(lstm_outputs, dropout_word_lstm)
           if self.config["whidden_layer_size"] > 0:
              lstm_outputs = tf.layers.dense(lstm_outputs, self.config["hidden_layer_size"], activation=tf.nn.relu, kernel_initializer=self.initializer)
           knowledge_output_vector_size = 2 * self.config["word_embedding_size"] 
           self.lstm_outputs = lstm_outputs         
           t_lstm_outputs = tf.transpose(lstm_outputs, [0, 2, 1])
           if self.config["sentence_composition"] == "attention":
                processed_tensor_1 = tf.expand_dims(processed_tensor_1, -1) #batch, Dim, 1
                processed_tensor_1 = tf.transpose(processed_tensor_1, [0,2,1])
                attention_weights = tf.matmul(processed_tensor_1,t_lstm_outputs) #batch, length_of_sentence, number of Knowledge
                if self.config["attention_activation"] == "sharp":
                    attention_weights = tf.exp(attention_weights)
                elif self.config["attention_activation"] == "soft":
                    attention_weights = tf.nn.softmax(attention_weights)
                    #pass
                elif self.config["attention_activation"] == "linear":
                    pass
                else:
                    raise ValueError("Unknown activation for attention: " + str(self.config["attention_activation"]))

                self.attention_weights_unnormalised = attention_weights 
                #attention_weights = tf.transpose(attention_weights, [0, 2, 1])# batch, 1,number of Knowledge
                self.attention_weights = attention_weights 
                #attention_weights = tf.squeeze(attention_weights)
                #attention_weights = tf.exp(attention_weights)
                sum_attention_weights = attention_weights
                
                self.attention_weights = tf.squeeze(attention_weights)

                
                #attention_weights = tf.matmul(weights_path,attention_weights)
                attention_weights = tf.transpose(attention_weights, [0, 2, 1])
                #attention_weights = tf.expand_dims(attention_weights, -1)
                
                processed_tensor_knowledge = tf.reduce_sum(lstm_outputs * attention_weights, axis=1)  # bs, d
                processed_tensor_knowledge = tf.layers.dense(processed_tensor_knowledge, self.config["hidden_layer_size"], activation=tf.nn.relu, kernel_initializer=self.initializer) 
                #processed_tensor_knowledge_att = tf.expand_dims(processed_tensor_knowledge, -1) #batch, Dim, 1
                #processed_tensor_knowledge_att = tf.transpose(processed_tensor_knowledge_att, [0,2,1]) #batch,1,Dim
                            ### attention over attention for the sentence
                #attention_weights = tf.matmul(processed_tensor_knowledge_att, processed_tensor_1)
                #attention_weights = tf.nn.softmax(attention_weights)
                #attention_weights = tf.transpose(attention_weights, [0, 2, 1])
                #processed_tensor_knowledge_sentence = tf.reduce_sum(attention_weights * tf.transpose(processed_tensor_1,[0,2,1]), axis=1)
                
           #if self.config["hidden_layer_size"] > 0:
                #processed_tensor_knowledge = tf.layers.dense(processed_tensor_knowledge, self.config["hidden_layer_size"], activation=tf.nn.relu, kernel_initializer=self.initializer) 
                #processed_tensor_knowledge_sentence = tf.layers.dense(processed_tensor_knowledge_sentence, self.config["hidden_layer_size"], activation=tf.nn.relu, kernel_initializer=self.initializer)        
                                
          
          
#####################################################################################################################################################
###############################################################CALCULATE SCORE################################################################
#####################################################################################################################################################  

          
         if self.config["sentence_composition"] == "attention":
              dense_input_sen_con = tf.concat([processed_tensor_sentence, processed_tensor_context],1)
              
              dense_input_sen_con = tf.layers.dense(dense_input_sen_con, self.config["hidden_layer_size"], activation=tf.nn.relu, kernel_initializer=self.initializer)      
              dense_input = tf.concat([processed_tensor_sentence, processed_tensor_knowledge],1) #,processed_tensor_knowledge,,processed_tensor_context
              dense_input = tf.layers.dense(dense_input, self.config["hidden_layer_size"], activation=tf.nn.relu, kernel_initializer=self.initializer) 
              
              
              final_score = (dense_input * dense_input_sen_con) + (dense_input * processed_tensor_knowledge)
              #final_score = dense_input_sen_con
              softmax_w = tf.get_variable('softmax_w', shape=[100, len(reiss)],initializer=tf.zeros_initializer, dtype=tf.float32)    
              
         elif self.config["sentence_composition"] == "last":   
              dense_input = tf.concat([processed_tensor_sentence, processed_tensor_context],2) #,processed_tensor_knowledge,processed_tensor_sentence,,
              dense_input = tf.reshape(dense_input,[self.batch_size, self.max_lengths[0] * dense_input.get_shape()[2]])#self.max_lengths[0] * dense_input.get_shape()[2]])
              #dense_input = tf.concat([dense_input, processed_tensor_knowledge],1)
              softmax_w = tf.get_variable('softmax_w',shape = [56*200,len(reiss)], initializer=tf.zeros_initializer, dtype=tf.float32)
              
              
         softmax_b = tf.get_variable('softmax_b', shape=[len(reiss)],initializer=tf.zeros_initializer, dtype=tf.float32)
          
          #if self.config["hidden_layer_size"] > 0:
          #    dense_input = tf.layers.dense(dense_input, self.config["hidden_layer_size"], activation=tf.nn.relu, kernel_initializer=self.initializer)
          
         self.sentence_scores = tf.matmul(final_score, softmax_w) + softmax_b
          
          
          
##################################################################################################################################################### 
###############################################################CALCULATE SCORE################################################################
#####################################################################################################################################################  

         if self.config["human_needs"] == "maslow":
                          
                    w = [3.3580651133263086, 2.4043071629811266, 2.948496008202039, 2.609976477765905, 2.3545068920496965]
         else:
                    #with belonging: 
                    w = [3.929112469627414, 4.352634266669815, 4.105348968927056, 4.009469417408209, 4.436903109491611, 3.4714643441750805, 4.533726764493145, 3.665643259544512, 5.264175448882736, 6.026320782448594, 3.7522367243231805, 3.8019798963053515, 7.896001211803761, 8.024995943144209, 15.275082043791086, 3.3076036095385644, 3.81662584588786, 8.618279130276653, 6.7344516295276895]
                    #without belonging class: 
                    #w = [3.929112469627414, 4.352634266669815, 4.105348968927056, 4.009469417408209, 4.436903109491611, 3.4714643441750805, 4.533726764493145, 3.665643259544512, 5.264175448882736, 6.026320782448594, 3.7522367243231805, 3.8019798963053515, 7.896001211803761, 8.024995943144209, 3.3076036095385644, 3.81662584588786, 8.618279130276653, 6.7344516295276895]
                       
         w = tf.convert_to_tensor(w, dtype=tf.float32)
         lossy = tf.nn.weighted_cross_entropy_with_logits(targets=self.sentence_labels,logits=self.sentence_scores, pos_weight=w)
          
         self.loss = tf.reduce_sum(lossy)
         regularizer = tf.nn.l2_loss(softmax_w)
         self.loss = tf.reduce_mean(self.loss+(0.01 * regularizer))
         self.sentence_scores = tf.nn.sigmoid(self.sentence_scores)
         self.train_op = self.construct_optimizer(self.config["opt_strategy"], self.loss, self.learningrate, self.config["clip"])
                 
              
    def construct_optimizer(self, opt_strategy, loss, learningrate, clip):
    
        optimizer = None
       
        if opt_strategy == "adadelta":
            optimizer = tf.train.AdadeltaOptimizer(learning_rate=learningrate)
        elif opt_strategy == "adam":
            optimizer = tf.train.AdamOptimizer(learning_rate=learningrate)
        elif opt_strategy == "sgd":
            optimizer = tf.train.GradientDescentOptimizer(learning_rate=learningrate)
        else:
            raise ValueError("Unknown optimisation strategy: " + str(opt_strategy))

        if clip > 0.0:
            grads, vs = zip(*optimizer.compute_gradients(loss))
            grads, gnorm  = tf.clip_by_global_norm(grads, clip)
            train_op = optimizer.apply_gradients(zip(grads, vs))
        else:
            train_op = optimizer.minimize(loss)
            
        return train_op


    def preload_word_embeddings(self, embedding_path):
        loaded_embeddings = set()
        embedding_matrix = self.session.run(self.word_embeddings)
        with codecs.open(embedding_path,encoding='utf-8',mode='r') as f:
            for line in f:
                line_parts = line.strip().split()
                if len(line_parts) <= 2:
                    continue
                w = line_parts[0]
                if self.config["lowercase"] == True:
                    w = w.lower()
                if self.config["replace_digits"] == True:
                    w = re.sub(r'\d', '0', w)
                if w in self.term2index and w not in loaded_embeddings:
                    word_id = self.term2index[w]
                    embedding = numpy.array(line_parts[1:])
                    embedding_matrix[word_id] = embedding
                    loaded_embeddings.add(w)            
        self.session.run(self.word_embeddings.assign(embedding_matrix))
        
    
    def translate2id(self, token, token2id, unk_token, lowercase=True, replace_digits=False):
    
        if lowercase == True:
            token = token.lower()
        if replace_digits == True:
            token = re.sub(r'\d', '0', token)
        token_id = None
        if token in token2id:
            token_id = token2id[token]
        elif unk_token != None:
            token_id = token2id[unk_token]
        else:
            raise ValueError("Unable to handle value, no UNK token: " + str(token))
        return token_id


    def create_input_dictionary_for_batch(self, batch, is_training, learningrate):
    
        max_length = 0
        max_lengths=[]
        sentence_max_lengths=[]
        knowledge_max_lengths=[]
        context_max_lengths=[]
        sentences_pad = []
        knowledge_pad = []
        context_pad = []
        id, sentences, knowledge, weight_per, context, label_distribution= zip(*batch)
        word_ids, sentence_lengths, sentence_classes, sentence_tokens = self.extract_input(sentences, label_distribution,0)
        word_ids_knowledge, knowledge_lengths, sentence_classes, knowledge_tokens = self.extract_input(knowledge,label_distribution,1)
        word_ids_context, context_lengths, sentence_classes, context_tokens = self.extract_input(context,label_distribution,0) 

        
        
        if self.config["sentence_composition"] == "last" or self.config["sentence_composition"] == "attention" :
            max_length_know = 0
            max_length_sent = 0
            max_length_context = 0
            max_length_sent = max(sentence_lengths)
            for i in range(len(knowledge_lengths)):
              if max_length_know < max(knowledge_lengths[i]):
                  max_length_know = max(knowledge_lengths[i])
            max_length_context = max(context_lengths)
            
            #print(max_length_know,max_length_context,max_length_sent)
            max_length = 56
            sentence_lengths=[]
            context_lengths=[]
            max_lengths=[]
            for i in range(len(sentences)):
                max_lengths.append(max_length)
                
            for i in range(len(sentences)):
                sentence_lengths.append(max_length_sent) 
               
            for i in range(len(sentences)):
                length = len(knowledge_lengths[i])
                knowledge_max_lengths.append([max_length_know]*length)  
            for i in range(len(sentences)):
                context_lengths.append(max_length_context)
            
            knowledge_pad=[]
            sentences_pad =  self._make_padding(word_ids, max_length_sent)
            for i in range(len(word_ids_knowledge)):
                knowledge_pad.append(self._make_padding(word_ids_knowledge[i], max_length_know))
            context_pad = self._make_padding(word_ids_context, max_length_context)
            input_dictionary = {self.word_ids: sentences_pad, self.batch_size: len(sentences), self.max_lengths: max_lengths, self.sentence_lengths: sentence_lengths, self.sentence_labels: sentence_classes, self.sentence_tokens: sentence_tokens, self.knowledge_tokens:knowledge_tokens, self.context_tokens:context_tokens, self.word_ids_knowledge: knowledge_pad, self.knowledge_max_lengths: knowledge_max_lengths, self.knowledge_lengths: knowledge_lengths, self.word_ids_context:  context_pad, self.context_lengths: context_lengths, self.learningrate: learningrate, self.is_training: is_training}#self.word_ids_knowledge: word_ids_knowledge,self.knowledge_lengths: knowledge_lengths,
                        
            #input_dictionary = {self.word_ids: sentences_pad, self.batch_size: len(sentences), self.max_lengths: max_lengths,self.weights_path:weight_per, self.sentence_lengths: sentence_lengths, self.sentence_labels: sentence_classes, self.sentence_tokens:sentence_tokens, ,self.word_ids_context:context_pad, self.context_tokens:context_tokens, self.context_lengths: context_lengths, self.learningrate: learningrate, self.is_training: is_training}
            #self.word_ids_knowledge: knowledge_pad, self.knowledge_max_lengths: knowledge_max_lengths, self.knowledge_tokens:knowledge_tokens, self.knowledge_lengths: knowledge_lengths,
            
        elif self.config["sentence_composition"] == "attention":
            input_dictionary = {self.word_ids: word_ids, self.batch_size: len(sentences), self.max_lengths: max_lengths, self.weights_path:weight_per, self.sentence_lengths: sentence_lengths, self.sentence_labels: sentence_classes,  self.word_ids_knowledge: word_ids_knowledge, self.knowledge_lengths: knowledge_lengths, self.knowledge_tokens:knowledge_tokens, self.word_ids_context: word_ids_context, self.context_lengths: context_lengths, self.context_tokens:context_tokens ,self.learningrate: learningrate, self.is_training: is_training}
        return input_dictionary
        
        
    def map_word2embedding(self, sent, word_ids):
    #map word to embeddings
       sent2embed=[[]]
       for i in range(len(sent)):
            x = sent[i].split(' ')
            x = [k for k in x if k]
            for j in range(len(x)):
                sent2embed[i].append(word_ids[x[j]])      
       return sent2embed
    
    
    def _make_padding(self, sequences,maximum):
    #padding the training data
       padded = keras.preprocessing.sequence.pad_sequences(sequences,maxlen=maximum)
       return(padded)
        
    def extract_input(self,X,y,l):
        
        if self.config["human_needs"] == "maslow":
            reiss=['physiological', 'love', 'spiritual growth', 'esteem', 'stability']
        elif self.config["human_needs"] == "reiss":
            #reiss=['status', 'approval', 'tranquility', 'competition', 'health', 'family', 'romance', 'food', 'indep', 'power', 'order', 'curiosity', 'serenity', 'honor', 'belonging', 'contact', 'savings', 'idealism', 'rest']
            reiss = ['status', 'approval', 'tranquility', 'competition', 'health', 'family', 'romance', 'food', 'indep', 'power', 'order', 'curiosity', 'serenity', 'honor', 'contact', 'savings', 'idealism', 'rest']
                
        sentence_lengths=[]
        max_length_count=[]
        sentence_list=[]
        max_1=0
        if l ==0:
          for i in range(len(X)):
            #x = X[i].split(' ')
            x = tknzr.tokenize(X[i])
            x = [k for k in x if k]
            sentence_lengths.append(len(x))  
            
        elif l ==1:
           max_1=0       
           for i in range(len(X)):
            if max_1<len(X[i]):
                max_1=len(X[i])
           all_lengths=[0]*max_1
           
           for i in range(len(X)):
              for j in range(len(X[i])):
                  #x = X[i][j].split(' ')
                  x = tknzr.tokenize(X[i][j])
                  x = [k for k in x if k]        
                  all_lengths[j]=len(x)
              sentence_lengths.append(all_lengths)
              all_lengths=[0]*max_1
        max_sentence_length = max(sentence_lengths)
        sentence_classes = [[]]
        sentence_labels = numpy.zeros((len(X), len(reiss)), dtype=numpy.float32)
        if l==0:
          word_ids = numpy.zeros((len(X),max_sentence_length), dtype=numpy.int32)
          sentence_list = [[' '] * max_sentence_length for i in range(len(X))]
          for i in range(len(X)):
            #x = X[i].split(' ')
            x = tknzr.tokenize(X[i])
            x = [k for k in x if k]
            count =0
            for j in range(len(x)): 
                 sentence_list[i][j]=x[j]
                 word_ids[i][j] = self.translate2id(x[j], self.term2index, self.UNK, lowercase=self.config["lowercase"], replace_digits=self.config["replace_digits"])
                 count+=1
            
            a = y[i].strip()
            a = ast.literal_eval(a)
            pos = []
            pos = [i for i, j in enumerate(a) if j == 1]
            sentence_classes[i]= [0]*(len(reiss))
            for l in pos:
                    sentence_classes[i][l]=1
            if i<len(X)-1:
                sentence_classes.append([]) 
                     
        elif l ==1:
         max_sentence_length = 0
         for i in range(len(X)):
            if max_sentence_length < max(sentence_lengths[i]):
                max_sentence_length = max(sentence_lengths[i])
         max_1=0       
         for i in range(len(X)):
            if max_1<len(X[i]):
                max_1=len(X[i])
         
         word_ids= numpy.zeros((len(X),max_1,max_sentence_length), dtype=numpy.int32)   
         sentence_list=[]  
         sentence_l = [[' '] * max_sentence_length for j in range(max_1)]
         for i in range(len(X)):
              sentence_list.append(sentence_l)   
         
         for i in range(len(X)):
           for j in range(len(X[i])):
             #x = X[i][j].split(' ')
             x = tknzr.tokenize(X[i][j])
             x = [k for k in x if k]
             
             for k in range(len(x)): 
                  word_ids[i][j][k]=self.translate2id(x[k], self.term2index, self.UNK, lowercase=self.config["lowercase"], replace_digits=self.config["replace_digits"]) 
                  sentence_list[i][j][k]=x[k]
                  
           a = y[i].strip()
           a = ast.literal_eval(a)
           pos = []
           pos = [i for i, j in enumerate(a) if j == 1]
           sentence_classes[i]= [0]*(len(reiss))
           for l in pos:
                    sentence_classes[i][l]=1
           if i<len(X)-1:
                sentence_classes.append([]) 
                
        return word_ids, sentence_lengths, sentence_classes, sentence_list


    def process_batch(self, data, batch, is_training, learningrate):
    
        feed_dict = self.create_input_dictionary_for_batch(batch, is_training, learningrate)
        cost, sentence_scores = self.session.run([self.loss, self.sentence_scores] + ([self.train_op] if is_training == True else []), feed_dict=feed_dict)[:2]
        token_scores=[]
        return cost, sentence_scores, token_scores
        


    def initialize_session(self):
        tf.set_random_seed(self.config["random_seed"])
        session_config = tf.ConfigProto()
        session_config.gpu_options.allow_growth = self.config["tf_allow_growth"]
        session_config.gpu_options.per_process_gpu_memory_fraction = self.config["tf_per_process_gpu_memory_fraction"]
        self.session = tf.Session(config=session_config)
        self.session.run(tf.global_variables_initializer())
        self.saver = tf.train.Saver(max_to_keep=1)


    def get_parameter_count(self):
        total_parameters = 0
        for variable in tf.trainable_variables():
            shape = variable.get_shape()
            variable_parameters = 1
            for dim in shape:
                variable_parameters *= dim.value
            total_parameters += variable_parameters
        return total_parameters


    def get_parameter_count_without_word_embeddings(self):
        shape = self.word_embeddings.get_shape()
        variable_parameters = 1
        for dim in shape:
            variable_parameters *= dim.value
        return self.get_parameter_count() - variable_parameters


    def save(self, filename):
        dump = {}
        dump["config"] = self.config
        dump["UNK"] = self.UNK
        dump["word2id"] = self.word2id

        dump["params"] = {}
        for variable in tf.global_variables():
            assert(variable.name not in dump["params"]), "Error: variable with this name already exists" + str(variable.name)
            dump["params"][variable.name] = self.session.run(variable)
        with codecs.open(filename, 'wb') as f:

            pickle.dump(dump, f, protocol=pickle.HIGHEST_PROTOCOL)


    @staticmethod
    def load(filename, new_config=None):
        with codecs.open(filename, 'rb') as f:
            dump = pickle.load(f)

            # for safety, so we don't overwrite old models
            dump["config"]["save"] = None

            # we use the saved config, except for values that are present in the new config
            if new_config != None:
                for key in new_config:
                    dump["config"][key] = new_config[key]

            labeler = MLTModel(dump["config"])
            labeler.UNK = dump["UNK"]
            labeler.term2index = dump["term2index"]
            labeler.singletons = dump["singletons"]

            labeler.construct_network()
            labeler.initialize_session()

            labeler.load_params(filename)

            return labeler


    def load_params(self, filename):
        with codecs.open(filename, 'rb') as f:
            dump = pickle.load(f)

            for variable in tf.global_variables():
                assert(variable.name in dump["params"]), "Variable not in dump: " + str(variable.name)
                assert(variable.shape == dump["params"][variable.name].shape), "Variable shape not as expected: " + str(variable.name) + " " + str(variable.shape) + " " + str(dump["params"][variable.name].shape)
                value = numpy.asarray(dump["params"][variable.name])
                self.session.run(variable.assign(value))

