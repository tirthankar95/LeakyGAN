import pandas as pd
from encode_decode import create_vocab
import numpy as np 
import sys 
sys.path.append("./")
from utils import get_arguments

punctuations = [',', '.', '?', ';', '!', "\"", "\'" "*"]

def create_frmt_data(filePath, pf, nf):
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
    for idx, text in enumerate(df["Questions"]):
        temp = []
        for word in text.split():
             temp.append(vocab[word])
        # PADDING & TRUNCATION
        seq_length = param_dict["model_params"]\
                               ["discriminator_params"]\
                               ["seq_len"]
        temp = temp + [len(vocab)-1] * (seq_length - len(temp)) \
               if seq_length > len(temp) else temp[:seq_length] 
        if df.loc[idx, "Valid"]: positive.append(temp)
        else: negative.append(temp)
    np.save(f"{pf}", positive)
    np.save(f"{nf}", negative)

if __name__ == "__main__":
    param_dict = get_arguments()
    create_frmt_data("./raw_data_datagencode/physics.csv",\
                     "./formatted_data/positive_corpus.npy",\
                     "./formatted_data/negative_corpus.npy")