__author__ = 'debjit'
import time
import collections
import numpy
import numpy as np
import random
import ast
import math
from random import randint
import codecs
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix


class MLTEvaluator(object):
    def __init__(self, config):
        self.config = config
        self.cost_sum = 0.0
        self.sentence_predicted={}
        self.sentence_correct={}
        
        if self.config["human_needs"] == "maslow":
          for i in range(5):
            self.sentence_predicted[i] = []
            self.sentence_correct[i] = []
        elif self.config["human_needs"] == "reiss":
            for i in range(18):
                self.sentence_predicted[i] = []
                self.sentence_correct[i] = []
        #f1  = codecs.open("/Users/debjit/Downloads/python_script/script/result/result.txt",encoding='utf-8',mode='w')
        #f1.close()        
        self.sentence_total = []
        self.X_test = []
        self.token_scores_list=[]
        self.start_time = time.time()
        self.count=0
        self.prec=0
        self.tot=0
        self.rec=0
        self.pos=0

    def append_data(self, cost, batch, sentence_scores, token_scores, name, epoch):
        assert(len(batch) == len(sentence_scores))
        self.cost_sum += cost
        sentence_pred_refined=[]
        sentence_cor_refined=[]
        
        #self.reiss=['physiological', 'love', 'spiritual growth', 'esteem', 'stability']
        if self.config["human_needs"] == "maslow":
            self.reiss=['physiological', 'love', 'spiritual growth', 'esteem', 'stability']
        elif self.config["human_needs"] == "reiss":
            self.reiss=['status', 'approval', 'tranquility', 'competition', 'health', 'family', 'romance', 'food', 'indep', 'power', 'order', 'curiosity', 'serenity', 'honor', 'belonging', 'contact', 'savings', 'idealism', 'rest']
            self.reiss = ['status', 'approval', 'tranquility', 'competition', 'health', 'family', 'romance', 'food', 'indep', 'power', 'order', 'curiosity', 'serenity', 'honor', 'contact', 'savings', 'idealism', 'rest']
                
        #self.labels=['0','1','2','3','4','5','6','7','8','9','10','11','12','13','14','15','16','17','18','19','20']
        id, sentences, lst2, weight_per, context, label_distribution = zip(*batch)
        max_1=0
        X = sentences
        y = label_distribution
        for i in range(len(X)):
            sentence_pred_refined=[]
            sentence_cor_refined=[]
            sentence_s=[]
            self.token_scores_list.append(token_scores)
            sentence_cor = []
            sentence_pred = []
            a=y[i].strip()
            a = ast.literal_eval(a)
            pos = []
            pos = [i for i, j in enumerate(a) if j == 1]
            sentence_cor=[0]*len(self.reiss)
            for l in pos:
                 sentence_cor[l]=1
            
            pos_pred = []
            lab = {'status': 456, 'approval': 346, 'tranquility': 411, 'competition': 441, 'health': 338, 'family': 706, 'romance': 397, 'food': 534, 'indep': 227, 'power': 179, 'order': 497, 'curiosity': 485, 'serenity': 93, 'honor': 99, 'belonging': 24, 'contact': 754, 'savings': 473, 'idealism': 87, 'rest': 129}
            
            if self.config["human_needs"] == "maslow":
                    w = [0.297790533016037, 0.41592023490047303, 0.3391559619610234, 0.38314521549097746, 0.4247173806866449]
            else:         
                    w = [0.25451040348937304, 0.22974592826635454, 0.24358465201591684, 0.24940955919459729, 0.22538242898763272, 0.28806287516043283, 0.2205690929219014, 0.27280341517037276, 0.18996327339588182, 0.16593872714384172, 0.2665077055287277, 0.26302085420592825, 0.12664638380565296, 0.12461065489438716, 0.06546609681919668, 0.30233368869116317, 0.2620115359427826, 0.11603244509532373, 0.1484901897008849]
            
            if name=='train':
              pos_pred = [j for j, h in enumerate(sentence_scores[i]) if h>=0.5]
              #print(sentence_scores)
            else:  
              pos_pred = [j for j, h in enumerate(sentence_scores[i]) if h>=0.5]
            sentence_pred = [0]*len(self.reiss)
            
            if len(pos_pred)>=1:
              for l in pos_pred: 
                 sentence_pred[l]=1  
            
            if name =='test':
                  f1  = codecs.open("./result_with_know_c2h/result_"+str(epoch)+".txt",encoding='utf-8',mode='a')
                  temp_cor=[]
                  temp_pred=[]
                  for l in pos:
                      temp_cor.append(self.reiss[l])
                  if pos_pred!=[]:
                    for l in pos_pred:
                      temp_pred.append(self.reiss[l])   
                  s = str(id[i])+'\t'+str(sentences[i])+'\t'+str(temp_cor)+'\t'+str(temp_pred)+'\n'
                  f1.write(s)
                  f1.close()
                  
            for w in range(len(sentence_pred)):
                self.sentence_predicted[w].append(sentence_pred[w])
                self.sentence_correct[w].append(sentence_cor[w])
                
            

    def get_results(self, name):
        assert(len(self.sentence_correct[0])==len(self.sentence_predicted[0]))
        print("GETTING RESULTs")
        f=[]
        p=[]
        r=[]
        acc=[]
        show={}
        f1=0
        p1 = 0
        r1 = 0
        f2 = []
        p2 = []
        r2 = []
        
        for i in range(len(self.sentence_predicted)):
          self.prec = 0
          self.tot = 0
          self.rec = 0
          self.pos = 0
          temp_p=0.0
          temp_f=0.0
          temp_r=0.0
          f2_sk=0
          
          f2.append(f1_score(np.array(self.sentence_correct[i]), np.array(self.sentence_predicted[i]),pos_label=1,average='micro')) 
          p2.append(precision_score(np.array(self.sentence_correct[i]), np.array(self.sentence_predicted[i]),pos_label=1,average='micro'))      
          r2.append(recall_score(np.array(self.sentence_correct[i]), np.array(self.sentence_predicted[i]),pos_label=1,average='micro'))
          for j in range(len(self.sentence_predicted[i])): 
              if self.sentence_predicted[i][j]!=0:
                     if self.sentence_predicted[i][j]==self.sentence_correct[i][j]:
                         self.prec+=1
                     self.pos+=1
              if self.sentence_correct[i][j]!=0:
                     if self.sentence_predicted[i][j]==self.sentence_correct[i][j]:
                         self.rec+=1
                     self.tot+=1
                     
          if self.pos>0:
                p.append(self.prec/self.pos)
                temp_p = self.prec/self.pos
          else: 
                p.append(0.0)
                temp_p = 0.0
          
          r.append(self.rec/self.tot)
          temp_r = self.rec/self.tot
          
          if temp_r==0.0 and temp_p==0.0:
              f.append(0.0)      
          else: 
              f.append((2*temp_r*temp_p)/(temp_p+temp_r))    
                                          
          
        
        p1 = sum(p) / len(p)
        r1 = sum(r) / len(r)
        if p1==0 and r1== 0:
             f1 = 0.0   
        else:
             f1 = (2*p1*r1)/(p1+r1)
             
        for i in range(len(self.reiss)):
           show[self.reiss[i]]= f[i]*100
        results = collections.OrderedDict()
        results[name + "_cost_sum"] = self.cost_sum
        results[name + "_tok_"+str()+"_p"] = p1*100
        results[name + "_tok_"+str()+"_r"] = r1*100
        results[name + "_tok_"+str()+"_f"] = f1*100
        
        results[name + "_tok_"+str()+"_f_sklearn"] = show
        
        results[name + "_time"] = float(time.time()) - float(self.start_time)

        return results
