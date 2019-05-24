__author__ = 'debjit'


import glob
import os
import subprocess
import sys
import argparse
import re
import csv
import gzip
import igraph
import ast
from igraph import *
from matplotlib import *
import matplotlib.pyplot as plt
import time
import codecs

def read_file(data,purpose,g):

   f = codecs.open(data,encoding='utf-8',mode='r')
   id_knowledge={}
   id_done=[]
   for line in f:
      #line = line.decode('utf-8')
      id = line.split('\t')[0].strip()
      print(id)
      if id not in id_done:
             concepts = line.split('\t')[-1]
             knowledge = make_sub_graph(id,purpose,concepts,g)
             id_knowledge[id]=knowledge
             id_done.append(id)
   return id_knowledge

def make_sub_graph(id,purpose,concepts,g):
      
      vertex = ['status', 'approval', 'safety', 'competition', 'health', 'family', 'love','romance','food', 'eating', 'independent', 'power', 'order', 'curiosity', 'calm','peace' ,'honor', 'belonging', 'social', 'money','save_money', 'idealism', 'rest']
      concepts = concepts.strip()
      if concepts !='No Knowledge for this sentence':
         concepts = ast.literal_eval(concepts)
      else:
         concepts = [] 
      concept_know_human=[]
      knowledge=[]
      nodes_1=[]
      nodes_2=[]
      edge_relation = []
      edges=[] 
      if concepts!=[]:
       for i in range(len(concepts)):
        for j in range(i+1,len(concepts)):
            if concepts[i]!=concepts[j]:
                try:
                    s = g.get_all_shortest_paths(concepts[i],concepts[j],mode='OUT')
                    k = g.get_all_shortest_paths(concepts[j],concepts[i],mode='OUT')
                    s = [list(x) for x in set(tuple(x) for x in s)]
                    k = [list(x) for x in set(tuple(x) for x in k)]
                    s.extend(k)
                    if len(s)==1:
                        if len(s[0])==1:
                            continue
                        else: 
                            y, w = extract_paths_between_concepts(s,g)
                            nodes_1.append(y)
                            nodes_2.append(w)
                            
                    elif s==[]: 
                            continue
                    else:                    
                        y, w = extract_paths_between_concepts(s,g)
                        nodes_1.append(y)
                        nodes_2.append(w)
                        
                except ValueError:
                            continue
       for i in concepts:
         for j in vertex:
            if i!=j:
                try:
                    s = g.get_all_shortest_paths(i,j,mode='OUT')
                   # k = g.get_all_shortest_paths(j,i,mode='OUT')
                    s = [list(x) for x in set(tuple(x) for x in s)]
                   # k = [list(x) for x in set(tuple(x) for x in k)]
                   # s.extend(k)
                    if len(s)==1:
                        if len(s[0])==1:
                            continue #"No Knowledge for "+concepts[i]+" and "+concepts[j]
                        else: 
                            y, w = extract_paths_between_concepts(s,g)
                            nodes_1.append(y)
                            nodes_2.append(w)
                            #edge_relation.append(v)
                    elif s==[]: 
                            continue
                    else:                    
                        y, w = extract_paths_between_concepts(s,g)
                        nodes_1.append(y)
                        nodes_2.append(w)
                        #edge_relation.append(v)
                except ValueError:
                            continue
       g_1=Graph(directed=True)
       flat_list_1=[]
       flat_list_2=[]
       nodes_1.extend(nodes_2)
       for sublist in nodes_1:
         for item in sublist:
             flat_list_1.append(item)
       
       unique_nodes = list(set(flat_list_1).union(set(flat_list_2)))
       
       for i in concepts:
        try:
         neighboring_nodes=g.neighbors(i)
         for j in range(len(neighboring_nodes)):
              unique_nodes.append(g.vs[neighboring_nodes[j]]["name"])
        except ValueError:
                            continue
                        
       for i in vertex:
        try:
         neighboring_nodes=g.neighbors(i)
         for j in range(len(neighboring_nodes)):
            unique_nodes.append(g.vs[neighboring_nodes[j]]["name"])
        except ValueError:
                            continue  
                 
       unique_nodes=list(set(unique_nodes))
       g_1 = g.subgraph(unique_nodes)   
       
       g_1.write_pickle('/home/mitarb/paul/Human_needs/script/subgraph/'+purpose+'/'+id) 
      return knowledge
      
          
def extract_paths_between_concepts(s,g):  
  
    edge_relation=[]
    nodes_1=[]
    nodes_2=[]
    edge=[]
    edge_relation=[]
    
    for i in range(len(s)):
      temp = ''
      for j in range(len(s[i])-1):
            nodes_1.append(g.vs[s[i][j]]["name"])
            nodes_2.append(g.vs[s[i][j+1]]["name"])
            
    return(nodes_1,nodes_2)
    
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("txtfile", help=".txt file containing the input text", nargs='?')
    parser.add_argument("graphpath", help=".txt file containing the input text", nargs='?')
    parser.add_argument("--purpose", help="train or dev or test", nargs='?')
    args = parser.parse_args()
    g = Graph(directed=True)
    g = read(args.graphpath, format="pickle")
    id_knowledge = read_file(args.txtfile,args.purpose,g)

    


if __name__ == '__main__':
    main()


