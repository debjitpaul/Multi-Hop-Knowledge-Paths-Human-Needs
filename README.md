# Ranking-and-Selecting-Multi-Hop-Knowledge-Paths-to-Better-Predict-Human-Needs
This directory contains the following parts of the 'Ranking and Selecting Multi-Hop Knowledge Paths to Better Predict Human Needs' experiment. We present a novel method to extract, rank, filter and select multi-hop relation paths from a commonsense knowledge resource to interpret the expression of sentiment in terms of their underlying human needs.

## Requirements 
~~~~
- python3+
- nltk
- igraph
- Tensorflow 
- Tensorflow-hub
~~~~

## Install
~~~~
- pip install tensorflow
- pip install tensorflow-hub
- pip install nltk
- pip install python-igraph
- pip install spacy 
- pip install sklearn
~~~~

## More about Data
[Modeling Naive Psychology of Characters in Simple Commonsense Stories](https://uwnlp.github.io/storycommonsense/)

## Run
### To construct subgraph per sentence 
~~~
python src/graph_model/
~~~
### To run the neural model: 
~~~
./src/neural_model/run_experiment.sh
~~~


## Reference

If you make use of the contents of this repository, please cite [the following paper](https://arxiv.org/abs/1904.00676):

```
@inproceedings{paulfrank:rankinghumanneeds,
  title={Ranking and Selecting Multi-Hop Knowledge Paths to Better Predict Human Needs},
  author={Paul, Debjit and Frank, Anette},
  booktitle={Proceedings of the 2019 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies (NAACL-HLT)},
  year={2019},
  address={Minneapolis, USA},
  note={to appear}
}
