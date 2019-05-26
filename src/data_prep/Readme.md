# Readme

## Data 
[Modeling Naive Psychology of Characters in Simple Commonsense Stories](https://uwnlp.github.io/storycommonsense/)
(CSV file)

## Data Prep:

~~~~
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
story+sentence id \t context \t sentence \t char \t emotion distribution 
story+sentence id \t context \t sentence \t char \t human need distribution 
~~~~

