import pandas as pd
from encode_decode import create_vocab
import numpy as np 
import sys 
sys.path.append("./")
from utils import get_arguments

punctuations = [',', '.', '?', ';', '!', "\"", "\'" "*"]

def create_frmt_data(filePath, pf):
    df = pd.read_csv(f"{filePath}")
    total_sentence = []
    for idx, text in enumerate(df["Questions"]):
        text_arr = text.split()
        # Case insensitive.
        text_arr = [word.lower() for word in text_arr]
        # Creating a gap between word and punctuations.
        new_text_arr = []
        for word in text_arr:
            if word[-1] in punctuations:
                new_text_arr.extend([word[:-1], word[-1]])
            elif word[0] in punctuations:
                new_text_arr.extend([word[0], word[1:]])
            else: new_text_arr.append(word)
        # New sentence to be added.
        df.loc[idx, "Questions"] = " ".join(new_text_arr)
        total_sentence.extend(new_text_arr)
    vocab, rev_vocab = create_vocab(" ".join(total_sentence), "./formatted_data/")
    positive, negative = [], []
    seq_length = param_dict["leak_gan_params"]\
                           ["discriminator_params"]\
                           ["seq_len"]
    for idx, text in enumerate(df["Questions"]):
        temp = []
        for word in text.split():
             temp.append(vocab[word])
        # PADDING & TRUNCATION
        temp = temp + [len(vocab)-1] * (seq_length - len(temp)) \
               if seq_length > len(temp) else temp[:seq_length] 
        positive.append(temp)
    np.save(f"{pf}", positive)

if __name__ == "__main__":
    param_dict = get_arguments()
    create_frmt_data("./rawdata_datagencode/physics.csv",\
                     "./formatted_data/positive_corpus.npy")