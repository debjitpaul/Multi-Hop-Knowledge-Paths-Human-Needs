import glob
import os
import sys
import argparse
import re
import csv
import gzip
import igraph
import ast
from igraph import *
import time
from heapq import nlargest
import operator
import collections
import itertools
from collections import defaultdict
import time
import codecs
import networkx as nx
start_time = time.time()

def read_file(input_path, graph_path, purpose):
   
    file = open(input_path,encoding='utf-8', mode='r')
    graph_file_name = graph_path.split('/')[-1]
    for lines in file:
      id = lines.split('\t')[0]
      if id.strip() == graph_file_name:
          concepts = lines.split('\t')[-1]
          concepts = concepts.strip()
          sentence = lines.split('\t')[2]
          concepts = ast.literal_eval(concepts)
          human = lines.split('\t')[1]
          return id.strip(), concepts, human
          
##############################################################################################################################################

def out_degree(graph):
  '''
  @ Get all the outdegree 
  '''
  list_of_outdegree=[]
  for v in graph.vs:
     if graph.outdegree(v['name'])==0:
           list_of_outdegree.append(v['name'])
  return list_of_outdegree
##############################################################################################################################################  
def get_top_need(concepts,vertex,g):
  s={}
  best_concepts =[]
  for i in vertex:
          s[i]=0
          try:
               neighbours = [] 
               neighbours=g.neighbors(i)
               neighbours.append(i)
          
               pr = g.personalized_pagerank(directed=True, damping=0.85, reset_vertices=neighbours, weights=None, implementation="prpack", niter=1000, eps=0.001)
               for j in concepts:
                  s[i]+=pr[g.vs.find(name=j).index]  
          except ValueError:
               continue
  best_concepts = nlargest(19, s, key=s.get) 
  
  return best_concepts

  
##############################################################################################################################################  
  
def get_top_concepts(concepts,vertex,g):
  max=0
  max_2=0
  count=0
  s={}
  best_concepts={}
  for i in vertex:
    for j in concepts:
      if i!=j:
          try:
               neighbours = [] 
               neighbours=g.neighbors(j)
               neighbours.append(j)
          
               pr = g.personalized_pagerank(directed=True, damping=0.85, reset_vertices=neighbours, weights=None, implementation="prpack", niter=1000, eps=0.001)  
               s[j]=pr[g.vs.find(name=i).index]  
               
          except ValueError:
               continue
    '''
    if len(concepts)>2:           
        best_concepts[i] = nlargest(3, s, key=s.get) 
    else:
    '''
    
    best_concepts[i] = nlargest(2, s, key=s.get)
    s={}
  
  return best_concepts  

##############################################################################################################################################
  
def path(concepts,g):
    g = g.simplify(multiple=True, loops=True, combine_edges="first")
    if concepts!=[]:
     for i in concepts:
        path = recurive_path(g.vs.find(name=i).index,g,'',[])
        
##############################################################################################################################################

def recurive_path(i,g,path,node_all):
        
        unique_nodes = get_neighours(i,g)
        score = nx.pagerank(g,vertices=unique_nodes, directed=True, damping=0.85, weights=None, arpack_options=None, implementation='prpack', niter=1000, eps=0.001)
        score.sort(reverse=True)
        count = 0
        node_1 = unique_nodes[score.index(score[count])]
        if node_1 in node_all: 
            count = count+1 
            node_1 = unique_nodes[score.index(score[count])]
            
        else: 
            node_all.append(node_1)
            node = g.vs.find(name=node_1).index
            path= path + g.vs[i]["name"]+'\t'+g[g.vs[i]["name"],g.vs[node]["name"]]+'\t'+g.vs[node]["name"]+'\t'
            if g.outdegree(node)!=0:
                  recurive_path(node,g,path,node_all) 
            else: 
                  return path
        return path    
        
##############################################################################################################################################

def get_neighours(i,g):
        unique_nodes=[]
        try:
           neighboring_nodes=g.neighbors(i,mode='out')
           for j in range(len(neighboring_nodes)):
              unique_nodes.append(g.vs[neighboring_nodes[j]]["name"])
        except ValueError:
                pass
        return unique_nodes
        
##############################################################################################################################################
        
        
def get_path(concepts, g):
    '''
    @ 
    '''
    vertex = ['status', 'approval', 'safety', 'competition', 'health', 'family', 'love', 'food', 'independent', 'power', 'order', 'curiosity', 'calm', 'honor', 'belonging', 'social', 'save_money','idealism', 'rest']
    #eating, money
    dict = {}
    scores = {}
    nodes_1=[]
    nodes_2=[]
    nodes_all=[]
    nodes=[]
    top_paths=[]
    temp_top_paths={}  ###top paths concepts to concepts
    edge_relation=[]
    concept2score={}
    
    best = get_top_concepts(concepts,vertex,g)
    need = get_top_need(concepts,vertex,g)
    human2concepts_paths = {}
    concepts2human_paths = {}
    for i in need:
            human2concepts_paths[i] = []
            concepts2human_paths[i] = []
##############################################################################################################################################        
    for i in need:
      concept=best[i]
      for j in range(len(concept)):
         if i!=concept[j]:
                try:
                    if j%2==0: 
                        s = g.get_all_shortest_paths(concept[j],i,mode='OUT')
                    else: 
                        s = g.get_all_shortest_paths(i,concept[j],mode='OUT')  
                    
                    s = [list(x) for x in set(tuple(x) for x in s)]
                    if len(s)==1:
                        if len(s[0])==1:
                            continue #"No Knowledge for "+concepts[i]+" and "+concepts[j]
                        else: 
                            dict,nodes = extract_paths_between_concepts_human_needs(s,g)
                            scores.update(dict)
                            top = nlargest(3, scores, key=scores.get)  
                            if j%2==0:
                               concepts2human_paths[i].extend(top)
                            else:
                               human2concepts_paths[i].extend(top) 
                            scores={}        
                    elif s==[]: 
                            continue
                    else:  
                        dict,nodes = extract_paths_between_concepts_human_needs(s,g)
                        scores.update(dict)  
                        top = nlargest(2, scores, key=scores.get)  
                        if j%2 ==0: 
                            concepts2human_paths[i].extend(top)
                        else:    
                            human2concepts_paths[i].extend(top)
                        
                        scores={}      
                except ValueError:
                        scores={} 
                        continue
      
      scores={} 
    temp_top_paths_refined = []  
    top_paths_refined = []  
    count=0 
    s=''    
    for j in need:  
        cpaths = concepts2human_paths[j]
        hpaths = human2concepts_paths[j]
        for x in cpaths:  
            s = x.replace('/r/','').replace('-->',' ')
            temp_top_paths_refined.append(s)
        for w in hpaths:  
            m = w.replace('/r/','').replace('-->',' ').split(' ')
            m = ' '.join(m)
            temp_top_paths_refined.append(m)

    for j in top_paths:        
          for y in j:
             top_paths_refined.append(y.replace('/r/','').replace('-->',' ')) 
    #print(temp_top_paths_refined)
   
    return temp_top_paths_refined    
##############################################################################################################################################

def extract_paths_between_concepts(s,g):  
    '''
    @ Extract paths between concepts
    '''
    edge_relation=[]
    nodes_1=[]
    nodes_2=[]
    edge=[]
    nodes=[]
    dict={}
    edge_relation=[]
    temp=''

    for i in range(len(s)):
      for j in range(len(s[i])-1):
               nodes.append(g.vs[s[i][j]]["name"])
               nodes.append(g.vs[s[i][j+1]]["name"])    
      temp=''
    return nodes
 
 
def extract_paths_between_concepts_human_needs(s,g):  
    '''
    @ Extract paths between concepts
    '''
    edge_relation=[]
    nodes_1=[]
    nodes_2=[]
    edge=[]
    nodes=[]
    dict={}
    edge_relation=[]
    temp=''
    for i in range(len(s)):
      nodes=[]
      temp=''
      for j in range(len(s[i])-1):
            if temp=='':
               temp=temp+g.vs[s[i][j]]["name"]+'-->'+g[g.vs[s[i][j]]["name"],g.vs[s[i][j+1]]["name"]]+'-->'+g.vs[s[i][j+1]]["name"]
               nodes.append(g.vs[s[i][j]]["name"])
               nodes.append(g.vs[s[i][j+1]]["name"])  
            else: 
               temp=temp+'-->'+g[g.vs[s[i][j]]["name"],g.vs[s[i][j+1]]["name"]]+'-->'+g.vs[s[i][j+1]]["name"]
               nodes.append(g.vs[s[i][j+1]]["name"])
      nodes_c_h = nodes[:]
      if len(nodes)>3:
            nodes_c_h.remove(g.vs[s[i][0]]["name"])
            nodes_c_h.remove(g.vs[s[i][-1]]["name"])
            #score_1 = closeness_path(nodes, g)
            score_1= personalized_page_rank(nodes,[g.vs[s[i][0]]["name"],g.vs[s[i][-1]]["name"]],g)
      else: 
            #score_1 = closeness_path(nodes, g)
            score_1= personalized_page_rank(nodes,[g.vs[s[i][0]]["name"],g.vs[s[i][-1]]["name"]],g)
      dict[temp] =  score_1 
      temp=''
      nodes=[]
    return dict, nodes
          
  
def closeness(nodes, graph):
    '''
    @ KPP score
    '''
    score = graph.closeness(vertices=nodes, mode='out', cutoff=None, weights=None, normalized=True)

    return score
def personalized_page_rank(nodes,list,g):
        '''
        @ PPR score
        '''
        score =0
        pr = g.personalized_pagerank(directed=True, damping=0.85, reset_vertices=nodes, weights=None, implementation="prpack", niter=1000, eps=0.001)
        score+=pr[g.vs.find(name=list[0]).index]  
        score+=pr[g.vs.find(name=list[-1]).index]  
                                    
        return score  

def closeness_path(nodes, graph):
    '''
    @ KPP score
    '''
    score = graph.closeness(vertices=nodes, mode='out', cutoff=None, weights=None, normalized=True)

    return sum(score)/len(nodes)     
    
def page_rank_path(nodes, graph):
    '''
    @ page_rank score
    '''
    score = graph.pagerank(vertices=nodes, directed=True, damping=0.85, weights=None, arpack_options=None, implementation='prpack', niter=1000, eps=0.001)
    return sum(score)/len(nodes) 

def page_rank(nodes, graph):
    '''
    @ page_rank score
    '''
    score = graph.pagerank(vertices=nodes, directed=True, damping=0.85, weights=None, arpack_options=None, implementation='prpack', niter=1000, eps=0.001)
    return score

def betweenness_path(nodes, graph):
    '''
    @ betweenness rank score
    '''
    score = graph.betweenness(vertices=nodes, directed=True, cutoff=None, weights=None, nobigint=True)
    return sum(score)/len(nodes)  
    
def betweenness(nodes, graph):
    '''
    @ betweenness rank score
    '''
    score = graph.betweenness(vertices=nodes, directed=True, cutoff=None, weights=None, nobigint=True)
    return score 

def main():
    '''
    @ 
    '''
    parser = argparse.ArgumentParser()
    parser.add_argument("--graph_path", help=".txt file containing the input text", nargs='?')
    parser.add_argument("--input_path", help=".txt file containing the input text", nargs='?')
    parser.add_argument("--output_path", help=".txt file containing the input text", nargs='?')
    parser.add_argument("--purpose", help="train or dev or test", nargs='?')
    
    args = parser.parse_args()
    
    id, concepts, human = read_file(args.graph_path, args.purpose)
    g = Graph(directed=True)
    g = read(args.graph_path, format="pickle")
    output = get_path(args.input_path,concepts,g)
    #f1  = codecs.open('/home/mitarb/paul/Human_needs/script/top_19_human_need_path/'+args.purpose+'/'+id,encoding='utf-8',mode='w')
    f1  = codecs.open(args.output_path,encoding='utf-8',mode='w')
    s = str(id)+'\t'+str(output)
    f1.write(s)
    f1.close()
    
    

if __name__ == '__main__':
    main()



