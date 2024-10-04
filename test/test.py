import sys
import torch
import torch.nn.functional as F
import logging
logging.basicConfig(
    level=logging.DEBUG, 
    format='%(asctime)s - %(filename)s - Line: %(lineno)d - %(message)s',  
)
sys.path.append("../")
from model.train_model import restore_checkpoint
from data_iter import Real_Dataset
from datagencode.encode_decode import tensor_to_text
from torch.utils.data import DataLoader
import pickle as pkl

def test_vocab():
    vocab = {}
    with open(f"../formatted_data/tokens.pkl", "rb") as file:
        vocab = pkl.load(file)
    print(vocab)

def test_positive_examples():
    positive_dataset = Real_Dataset("../formatted_data/positive_corpus.npy")
    data_loader = DataLoader(positive_dataset, shuffle = True, batch_size = 4)
    for sample in data_loader:
        for sentence in sample:
            logging.debug(tensor_to_text(sentence, ".././formatted_data/"))


def test_discriminator(positive, negative):
    discriminator = restore_checkpoint(prefix = "../")["model_dict"]["discriminator"]
    logging.debug("Discriminator Loaded.")
    discriminator = discriminator.eval()
    # Check all positive examples.
    if positive == True:
        mean_batch_pred = []
        logging.debug("Positive Examples.")
        positive_dataset = Real_Dataset("../formatted_data/positive_corpus.npy")
        data_loader = DataLoader(positive_dataset, shuffle = True, batch_size = 4)
        for sample in data_loader:
            pred = F.softmax(discriminator(sample)['pred'], dim = 1)
            mean_batch_pred.append(torch.mean(pred, dim = 0))
        mean_batch_pred = torch.stack(mean_batch_pred, dim = 0)
        mean_batch_pred = torch.mean(mean_batch_pred, dim = 0)
        logging.debug(f"Class 0: {round(mean_batch_pred[0].item(), 5)}")
        logging.debug(f"Class 1: {round(mean_batch_pred[1].item(), 5)}")
        logging.debug("-----------------------------------------------")
        logging.debug("\n")
    # Check all negative examples.
    if negative == True:
        mean_batch_pred = []
        logging.debug("Negative Examples.")
        negative_dataset = Real_Dataset("../formatted_data/negative_corpus.npy")
        data_loader = DataLoader(negative_dataset, shuffle = True, batch_size = 4)
        for sample in data_loader:
            pred = F.softmax(discriminator(sample)['pred'], dim = 1)
            mean_batch_pred.append(torch.mean(pred, dim = 0))
        mean_batch_pred = torch.stack(mean_batch_pred, dim = 0)
        mean_batch_pred = torch.mean(mean_batch_pred, dim = 0)
        logging.debug(f"Class 0: {round(mean_batch_pred[0].item(), 5)}")
        logging.debug(f"Class 1: {round(mean_batch_pred[1].item(), 5)}")
        logging.debug("-----------------------------------------------")
        logging.debug("\n")
    assert True

if __name__ == '__main__':
    # test_vocab()
    # test_positive_examples()
    test_discriminator(True, True)