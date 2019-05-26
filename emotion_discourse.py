#python script to read the "Modeling Naive Psychology of Characters in Simple Commonsense Stories emotion"
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
from heapq import nlargest


def read_file(data):
    """ 
    read line by line 
    @param text: path of the input file
    @type text: str
    
    @return: print ----> story+sentence id \t context \t sentence \t char \t emotion distribution 
    """
    graph = []
    clean_graph=[]
    n=0
    a=0
    out=[]
    count_out=[]
    input_sentence=[]
    input_char=[]
    tags=[]
    tag=''
    flag=False
    emotion_p = ["joy","trust","fear","surprise","sadness","disgust","anger","anticipation"]
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
    distribution= [0] * len(emotion_p)
    count_distribution= [0] * len(emotion_p)
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
             if line[-3]=='yes':
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
                
                if s == line[-5]:
                  if c == line[2]:
                   context.append(con)
                   distribution= [0] * len(emotion_p)
                   count_distribution = [0] * len(emotion_p)
                   for i in range(len(line[-1])):
                      if line[-1][i].strip().split(':')[0] in emotion_p:                            
                             pos= emotion_p.index(line[-1][i].strip().split(':')[0])
                             
                             distribution[pos]=distribution[pos]+1
                             count_distribution[pos]= count_distribution[pos]+int(line[-1][i].strip().split(':')[-1])
                  else: 
                      story_ids.append(w+'__sent'+str(line_num))
                      context.append(con)
                      count=count+1
                      input_sentence.append(s)
                      input_char.append(c)
                      
                      out.append(distribution)
                      count_out.append(count_distribution)
                      s = line[-5]
                      c = line[2]
                      con = line[4]
                      line_num=line[1]
                      if con == '':
                         con = "No Context"
                         line_num=1
                      distribution= [0] * len(emotion_p)
                      count_distribution = [0] * len(emotion_p)
                      for i in range(len(line[-1])):
                        if line[-1][i].strip().split(':')[0] in emotion_p:                      
                                pos= emotion_p.index(line[-1][i].strip().split(':')[0])
                                distribution[pos]=distribution[pos]+1  
                                count_distribution[pos]= count_distribution[pos]+int(line[-1][i].strip().split(':')[-1])      
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
                        if line[-1][i].strip().split(':')[0] in emotion_p:                      
                                pos= emotion_p.index(line[-1][i].strip().split(':')[0])
                                distribution[pos]=distribution[pos]+1
                                count_distribution[pos]= count_distribution[pos]+int(line[-1][i].strip().split(':')[-1])
                              
                  else: 
                           story_ids.append(w+'__sent'+str(line_num))
                           context.append(con)
                           input_sentence.append(s)
                           input_char.append(c)
                           out.append(distribution)
                           count_out.append(count_distribution)
                           s = line[-5]
                           c = line[2]
                           con = line[4]
                           line_num=line[1]
                           w=line[0]
                           if con == '':
                                con = "No Context"
                           line_num=1
                           distribution= [0] * len(emotion_p)
                           count_distribution = [0] * len(emotion_p)
                           for i in range(len(line[-1])):
                              if line[-1][i].strip().split(':')[0] in emotion_p:                      
                                pos= emotion_p.index(line[-1][i].strip().split(':')[0])
                                distribution[pos]=distribution[pos]+1 
                                count_distribution[pos]= count_distribution[pos]+int(line[-1][i].strip().split(':')[-1])
                else:
                           
                           story_ids.append(w+'__sent'+str(line_num))
                           context.append(con)
                           input_sentence.append(s)
                           input_char.append(c)
                      
                           out.append(distribution)
                           count_out.append(count_distribution)
                           s = line[-5]
                           c = line[2]
                           con = line[4]
                           line_num=line[1]
                           w=line[0]
                           if con == '':
                                con = "No Context"
                           line_num=1
                           distribution= [0] * len(emotion_p)
                           count_distribution = [0] * len(emotion_p)
                           for i in range(len(line[-1])):
                             if line[-1][i].strip().split(':')[0] in emotion_p:                      
                                pos= emotion_p.index(line[-1][i].strip().split(':')[0])
                                distribution[pos]=distribution[pos]+1  
                                count_distribution[pos]= count_distribution[pos]+int(line[-1][i].strip().split(':')[-1])           
                     
             else:
                if indicator==True:
                    indicator=False
                    story_ids.append(w+'__sent'+str(line_num))
                    context.append(con)
                    n=1
                    input_sentence.append(s)
                    input_char.append(line[2])
                    out.append(distribution)
                    count_out.append(count_distribution)
                    s = line[-5]
                    c = line[2]
                    con = line[4]
                    line_num=line[1]
                    w=line[0]
                    if con == '':
                       con = "No Context"
                       line_num=1
                    distribution= [0] * len(emotion_p)
                    count_distribution = [0] * len(emotion_p)
                    for i in range(len(line[-1])):
                        if line[-1][i].strip().split(':')[0] in emotion_p:                      
                                pos= emotion_p.index(line[-1][i].strip().split(':')[0])
                                distribution[pos]=distribution[pos]+1 
                                count_distribution[pos]= count_distribution[pos]+int(line[-1][i].strip().split(':')[-1])
        else:
        # No more lines to be read from file
            story_ids.append(w+'__sent'+str(line_num))
            context.append(con)
            n=1
            count_out.append(count_distribution)
            input_sentence.append(s)
            input_char.append(line[2])
            out.append(distribution)
                                
    count=0
    
    
    for i in range(len(out)):
        if sum(count_out[i])!=0:
              count=count+1
              print(story_ids[i],'\t',context[i].replace('|',' '),'\t',input_sentence[i],'\t',input_char[i],'\t',count_out[i])
    return

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("txtfile", help=".txt file containing the input text", nargs='?')
    args = parser.parse_args()
    read_file(args.txtfile)


if __name__ == '__main__':
    main()