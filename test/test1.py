import pickle 

def test1():
    with open("./formatted_data/tokens.pkl", "rb") as file:
        vocab = pickle.load(file)
    for k, v in vocab.items():
        print(f"[{k} -> {v}]", end = ", ")

if __name__ == '__main__':
    test1()