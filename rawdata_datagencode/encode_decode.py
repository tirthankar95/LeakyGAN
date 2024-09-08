import numpy as np
import pickle

def create_vocab(corpus, filePath):
    reverse_vocab = [""] + list(set(corpus.strip().split())) + ["<R>"]
    vocab = { word: idx for idx, word in enumerate(reverse_vocab)}
    with open(f"{filePath}/chars.pkl", "wb") as f:
        pickle.dump(reverse_vocab, f)
    np.save(f"{filePath}/vocab.npy", vocab)
    return vocab, reverse_vocab

def tensor_to_text(input_x, filepath, sen_len = 15):
    with open(filepath, "rb") as f:
        vocab = pickle.load(f)
    text = [""] # For aligning.
    for index, x in enumerate(input_x):
        if (index+1) % sen_len == 0:
            text.append("\n")
        text.append(vocab[-1 if x >= len(vocab) else x]) 
    text_ = " ".join(text) # More efficient than functools's reduce, memory is allocated once.
    return text_