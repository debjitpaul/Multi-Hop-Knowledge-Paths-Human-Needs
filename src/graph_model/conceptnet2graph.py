import os
import glob
import os
import subprocess
import sys
import argparse
import re
import csv
import gzip
import igraph
from igraph import *
import matplotlib
from matplotlib import *


def ontology_create(data):
    graph = []
    clean_graph=[]
    n=0
    a=0
    words=[]
    new_tags=[]
    tags=[]
    tag=''
    flag=False
    count=0
    nodes_1=[]
    nodes_2=[]
    vertices=[]
    relation=[]
    g = Graph()
    edges=[]
    with open(data, 'r') as csvfile:
       read = csv.reader(csvfile, delimiter='\t', quotechar='\t') 
       for line in read: 
             if '/c/en/' in line[2] and '/c/en/' in line[3]:
                      nodes_1.append(line[2])#.replace('/c/en/','').replace('/n','').replace('/v','').replace('/a','').replace('/s','').replace('/r',''))
                      nodes_2.append(line[3])#.replace('/c/en/','').replace('/n','').replace('/v','').replace('/a','').replace('/s','').replace('/r',''))
                      relation.append(line[1])
                      count=count+1
                      
                      
                 
    edges=zip(nodes_1,nodes_2)
    unique_nodes = list(set(nodes_1).union(nodes_2))
    vertex = unique_nodes[:]
    print(unique_nodes)
    print(len(vertex))
    for i in range(len(unique_nodes)):
         g.add_vertices(unique_nodes[i])
    g.add_edges(edges) 
    g.es['weight'] = relation
    print(summary(g))
    g.write_pickle('concept_graph_full')
    return
          
       

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("txtfile", help=".txt file containing the input text", nargs='?')
    args = parser.parse_args()
    ontology_create(args.txtfile)
    


if __name__ == '__main__':
    main()



    