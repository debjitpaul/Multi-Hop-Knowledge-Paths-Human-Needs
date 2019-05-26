#python script to read the "Modeling Naive Psychology of Characters in Simple Commonsense Stories Human needs"
#!/usr/bin/env python3
import glob
import os
import subprocess
import sys
import argparse
import re
import csv
import json
import gzip
import random
import operator
import collections
import itertools
from collections import defaultdict



def read_file(data):
    graph = []
    clean_graph=[]
    n=0
    a=0
    out=[]
    input_sentence=[]
    input_char=[]
    tags=[]
    tag=''
    flag=False
    #human_needs=["curiosity","serenity","idealism","independence","competition","honor","approval","power","status","romance","belonging","family","social contract","health","savings","order","safety","food","rest","none"]
    #human_needs=['status', 'approval', 'tranquility', 'competition', 'health', 'family', 'romance', 'food', 'indep', 'power', 'order', 'curiosity', 'serenity', 'honor', 'belonging', 'contact', 'savings', 'idealism', 'rest']
    human_needs = ['status', 'approval', 'tranquility', 'competition', 'health', 'family', 'romance', 'food', 'indep', 'power', 'order', 'curiosity', 'serenity', 'honor', 'belonging', 'contact', 'savings', 'idealism', 'rest']
    #human_needs = ['status', 'approval', 'tranquility', 'competition', 'health', 'family', 'romance', 'food', 'indep', 'power', 'order', 'curiosity', 'serenity', 'honor', 'contact', 'savings', 'idealism', 'rest']
    #super_class = {'physiological':['food','rest'] , 'love':['love','belonging', 'social','family'], 'spiritual growth':['curiosity','idealism','independent','competition','calm'],'esteem':['power','honor','approval','competition','status'],'stability':['health','order','save_money','safety']}
  
    #human_needs=['[]']
    #human_needs =['physiological', 'love', 'spiritual growth', 'esteem', 'stability']
    count_line=0 
    count_story=0
    context=[]
    con=''
    classification=[]
    n=[]
    motivation={}
    line_num=''
    story_ids=[]
    ex_story_ids=[]
    w=''
    s=''
    temp_motivation=[]
    indicator = False
    distribution= [0] * len(human_needs)
    
    with open(data, newline='') as csvfile:
        lines= csv.reader(csvfile)
        
        for line in lines:
             if count_story==0:
                    count_story=count_story+1
                    story_id=line[0]
                    ex_story_ids=line[0]
                    
             elif story_id!=line[0]:
                 ex_story_ids=story_id
                 story_id=line[0]
                 count_story=count_story+1
             else:
                 story_id=line[0]
                 count_story=count_story 

             line[-1]=line[-1].replace("[", "")
             line[-1]=line[-1].replace("]", "")
             line[-1]=line[-1].replace('"', "")
             line[-1]=line[-1].split(",")
             if line[-4]=='yes':
               if indicator ==False:
                     s = line[-5]
                     c = line[2]
                     con = line[4]
                     line_num=line[1]
                     indicator=True
               if count_line==0:
                count_line=count_line+1
                s = line[-5]
                c = line[2]
                con = line[4]
                
                line_num=line[1]
                w=line[0]
                if con == '':
                       con = "No Context"
                       line_num=1
                line[-3]=line[-3].replace("[", "")
                line[-3]=line[-3].replace("]", "")
                line[-3]=line[-3].replace('"', "")
                line[-3]=line[-3].replace(",", "")
                #temp_motivation.append(line[-3])
                if s == line[-5]:
                  if c == line[2]:
                   #context.append(con)
                   distribution= [0] * len(human_needs)
                   
                   for i in range(len(line[-1])):
                      classification.append(str(line[-1][i]))
                      
                      if line[-1][i] in human_needs:                            
                             pos= human_needs.index(line[-1][i])
                             distribution[pos]=distribution[pos]+1
                  else: 
                      story_ids.append(w+'__sent'+str(line_num))
                      context.append(con)
                      #motivation[count]=temp_motivation
                      #temp_motivation=[]
                      count=count+1
                      input_sentence.append(s)
                      input_char.append(c)
                      
                      out.append(distribution)
                      s = line[-5]
                      c = line[2]
                      con = line[4]
                      line_num=line[1]
                      if con == '':
                         con = "No Context"
                         line_num=1
                      distribution= [0] * len(human_needs)
                      
                      for i in range(len(line[-1])):
                        if line[-1][i].strip() in human_needs:                      
                                pos= human_needs.index(line[-1][i].strip())
                                distribution[pos]=distribution[pos]+1          
               else:
                if s == line[-5]:
                  if c == line[2]:
                    con = line[4]
                    line_num=line[1]
                    w=line[0]
                    if con == '':
                       con = "No Context"
                       line_num=1
                    line[-3]=line[-3].replace("[", "")
                    line[-3]=line[-3].replace("]", "")
                    line[-3]=line[-3].replace('"', "")
                    line[-3]=line[-3].replace(",", "")
                    temp_motivation.append(line[-3])    
                    

                    for i in range(len(line[-1])):
                        if line[-1][i].strip() in human_needs:                      
                                pos= human_needs.index(line[-1][i].strip())
                                distribution[pos]=distribution[pos]+1  
                              
                  else: 
                           story_ids.append(w+'__sent'+str(line_num))
                           context.append(con)
                           input_sentence.append(s)
                           input_char.append(c)
                           out.append(distribution)
                           s = line[-5]
                           c = line[2]
                           con = line[4]
                           line_num=line[1]
                           w=line[0]
                           if con == '':
                                con = "No Context"
                           line_num=1
                           distribution= [0] * len(human_needs)
                           for i in range(len(line[-1])):
                              if line[-1][i].strip() in human_needs:                      
                                pos= human_needs.index(line[-1][i].strip())
                                distribution[pos]=distribution[pos]+1 
                else:
                           
                           story_ids.append(w+'__sent'+str(line_num))
                           context.append(con)
                           input_sentence.append(s)
                           input_char.append(c)
                      
                           out.append(distribution)
                           s = line[-5]
                           c = line[2]
                           con = line[4]
                           line_num=line[1]
                           w=line[0]
                           if con == '':
                                con = "No Context"
                           line_num=1
                           distribution= [0] * len(human_needs)
                           for i in range(len(line[-1])):
                             if line[-1][i].strip() in human_needs:                      
                                pos= human_needs.index(line[-1][i].strip())
                                distribution[pos]=distribution[pos]+1                  
                     
             else:
                if indicator==True:
                    indicator=False
                    story_ids.append(w+'__sent'+str(line_num))
                    context.append(con)
                    n=1
                    input_sentence.append(s)
                    input_char.append(line[2])
                    out.append(distribution)
                    s = line[-5]
                    c = line[2]
                    con = line[4]
                    line_num=line[1]
                    w=line[0]
                    if con == '':
                       con = "No Context"
                       line_num=1
                    distribution= [0] * len(human_needs)
                    for i in range(len(line[-1])):
                        if line[-1][i].strip() in human_needs:                      
                                pos= human_needs.index(line[-1][i].strip())
                                distribution[pos]=distribution[pos]+1 
        else:
        # No more lines to be read from file
            story_ids.append(w+'__sent'+str(line_num))
            context.append(con)
            n=1
            input_sentence.append(s)
            input_char.append(line[2])
            out.append(distribution)
                                
    count=0
    for i in range(len(out)):
        if sum(out[i])!=0:
           if 2 in out[i] or 3 in out[i]:
              for n, a in enumerate(out[i]):
                 if a==1:
                     out[i][n]=0
                 if a==2 or a==3:
                    out[i][n]=1  
              count=count+1
              print(story_ids[i],'\t',context[i],'\t',input_sentence[i],'\t',input_char[i],'\t',out[i])
    
    return

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("txtfile", help=".txt file containing the input text", nargs='?')
    args = parser.parse_args()
    read_file(args.txtfile)
    


if __name__ == '__main__':
    main()
