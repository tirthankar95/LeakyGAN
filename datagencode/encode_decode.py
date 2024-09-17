import numpy as np
import traceback
import pickle

def add_to_vocab(corpus, filePath):
    '''
        vocab -> [word to index]
        reverse_vocab -> [index to word]
    '''
    vocab, reverse_vocab = {}, []
    try:
        with open(f"{filePath}/tokens.pkl", "rb") as f:
            vocab = pickle.load(f)
        reverse_vocab = np.load(f"{filePath}/rvocab.npy")
    except Exception as e: 
        traceback.print_exc()
        print(f'{e}.\n[TM] first creation of files.')
    new_words = ["<R>"] + list(set(corpus.strip().split()))
    indx = len(vocab)
    for word in new_words:
        if word not in vocab:
            vocab[word] = indx 
            reverse_vocab.append(word)
            indx += 1
    np.save(f"{filePath}/rvocab.npy", reverse_vocab)
    with open(f"{filePath}/tokens.pkl", "wb") as f:
        pickle.dump(vocab, f)
    return vocab, reverse_vocab

def tensor_to_text(input_x, filepath, sen_len = 15):
    with open(filepath, "rb") as f:
        vocab = pickle.load(f)
    # For aligning.
    text = [""] 
    for index, x in enumerate(input_x):
        if (index+1) % sen_len == 0:
            text.append("\n")
        text.append(vocab[0] if x not in vocab else vocab[x]) 
    # More efficient than functools's reduce, memory is allocated once.
    text_ = " ".join(text) 
    return text_