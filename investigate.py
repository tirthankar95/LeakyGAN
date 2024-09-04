import numpy as np 
import pickle 

positive_filepath, negative_filepath = "./data/train_corpus.npy", "./data/gen_corpus.npy"

pos_data = np.load(positive_filepath, allow_pickle=True)
neg_data = np.load(negative_filepath, allow_pickle=True)

print(pos_data.shape, neg_data.shape)

corpus = np.load("./data/corpus.npy")
print(corpus.shape)

# Chinese Poem Dataset.
with open("./data/chars.pkl", "rb") as file:
    data = pickle.load(file)

print(data)
