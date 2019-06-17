# Readme

## Data 
[Modeling Naive Psychology of Characters in Simple Commonsense Stories](https://uwnlp.github.io/storycommonsense/)
(CSV file)

## Data Prep:

~~~~
1. It contains narrative stories where each sentence is annotated with a character and a set of human need categories
from two inventories: Maslow’s (with five coarsegrained) and Reiss’s (with 19 fine-grained) categories (Reiss’s labels are considered as subcategories of Maslow’s). 
2. Since no training data is available, similar to prior work we use a portion of the devset as training data, by performing a random split, using 80% of the data to train the classifier, and 20% to tune parameters. 

> Emotion: 
We considered the instances if affected==yes -selected by Mturk workers and we take the "majority label" for Plutchik to be categories

> Human needs: 
We considered the instances if affected==yes and we take the "majority label" for Maslow/Reiss to be categories voted on by >=2 workers

~~~~
## Data Distribution:
~~~
In our experiment we read the data in the following way: 

> It is a multi-label classifiaction task. 

> Emotion distribution: ["joy","trust","fear","surprise","sadness","disgust","anger","anticipation"]

> Human need distribution: ['status', 'approval', 'tranquility', 'competition', 'health', 'family', 'romance', 'food', 'indep', 'power', 'order', 'curiosity', 'serenity', 'honor', 'belonging', 'contact', 'savings', 'idealism', 'rest']

~~~

## Sample Example: 
~~~~
story+sentence_id \t context \t sentence \t char \t emotion distribution 
story+sentence_id \t context \t sentence \t char \t human need distribution 
~~~~

## Where can we find the sample?

~~~~
/src/data_prep
~~~~




