import nltk 
import random
import pandas as pd 
from nltk.corpus import wordnet as wn

# nltk.download('wordnet')
all_synsets = list(wn.all_synsets())
sentences = []
suffix = "The {x} loves to swim and {y}."
M = 2000
for _ in range(M):
    random_synset = random.choice(all_synsets)
    x = random_synset.lemmas()[0].name()
    random_synset = random.choice(all_synsets)
    y = random_synset.lemmas()[0].name()
    sentences.append(suffix.format(x = x, y = y))

df = pd.DataFrame({"Questions": sentences})
df.to_csv('sanity.csv', index = False)


