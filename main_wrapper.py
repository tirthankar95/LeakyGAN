import numpy as np 
import pickle 
import torch
from utils import recurrent_func, get_arguments
from rawdata_datagencode.encode_decode import tensor_to_text
from main import restore_checkpoint

#########################################################################

# positive_filepath, negative_filepath = "./formatted_data/positive_corpus.npy", \
#                                        "./formatted_data/negative_corpus.npy"

# pos_data = np.load(positive_filepath, allow_pickle = True)
# neg_data = np.load(negative_filepath, allow_pickle = True)

# print(pos_data.shape, neg_data.shape)
# print(pos_data, neg_data)

# Chinese Poem Dataset.
# with open("./formatted_data/chars.pkl", "rb") as file:
#     data = pickle.load(file)
# print(data)

# Generate random chinese poem
# from encode_decode import tensor_to_text
# print(tensor_to_text(torch.randint(1, 20, size = (20,)),\
#                      "./formatted_data/chars.pkl"))

#########################################################################

def get_sentence(gen_tokens):
    for token in gen_tokens:
        print(tensor_to_text(token, "./formatted_data/chars.pkl"))
        print(f'------------------\n')


if __name__ == '__main__':
    # Load generator model. 
    param_dict = get_arguments()
    restore_checkpoint_path = param_dict["train_params"]["checkpoint_path"]
    model = restore_checkpoint(restore_checkpoint_path)["model_dict"]
    gen_tokens = recurrent_func("gen")(model)

    get_sentence(gen_tokens)

#########################################################################