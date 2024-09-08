import numpy as np 
import pickle 
import torch
from utils import recurrent_func
from rawdata_datagencode.encode_decode import tensor_to_text

positive_filepath, negative_filepath = "./formatted_data/positive_corpus.npy", \
                                       "./formatted_data/negative_corpus.npy"

pos_data = np.load(positive_filepath, allow_pickle = True)
neg_data = np.load(negative_filepath, allow_pickle = True)

print(pos_data.shape, neg_data.shape)
# print(pos_data, neg_data)

# Chinese Poem Dataset.
# with open("./formatted_data/chars.pkl", "rb") as file:
#     data = pickle.load(file)
# print(data)

# Generate random chinese poem
# from encode_decode import tensor_to_text
# print(tensor_to_text(torch.randint(1, 20, size = (20,)),\
#                      "./formatted_data/chars.pkl"))

# Load generator model. 
MODEL_PATH = "./"
def restore_checkpoint(filename):
    checkpoint = torch.load(MODEL_PATH + filename, weights_only = False)
    return checkpoint

model = restore_checkpoint("checkpoint0.pth.tar")["model_dict"]
gen_tokens = recurrent_func("gen")(model)

def get_sentence(gen_tokens):
    for token in gen_tokens:
        print(tensor_to_text(token, "./formatted_data/chars.pkl"))
        print(f'------------------\n')

get_sentence(gen_tokens)
