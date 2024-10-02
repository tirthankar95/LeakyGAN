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
from torch.utils.data import DataLoader

def test_discriminator(positive, negative):
    discriminator = restore_checkpoint(prefix = "../")["model_dict"]["discriminator"]
    logging.debug("Discriminator Loaded.")
    # Check all positive examples.
    if positive == True:
        logging.debug("Positive Examples.")
        positive_dataset = Real_Dataset("../formatted_data/positive_corpus.npy")
        data_loader = DataLoader(positive_dataset, shuffle = True, batch_size = 4)
        discriminator = discriminator.eval()
        for sample in data_loader:
            pred = F.softmax(discriminator(sample)['pred'], dim = 1)
            logging.debug(f"Class 0: {pred[:, 0]}")
            logging.debug(f"Class 1: {pred[:, 1]}")
            logging.debug("-----------------------------------------------")
        logging.debug("\n")
    # Check all negative examples.
    if negative == True:
        logging.debug("Negative Examples.")
        negative_dataset = Real_Dataset("../formatted_data/negative_corpus.npy")
        data_loader = DataLoader(negative_dataset, shuffle = True, batch_size = 4)
        for sample in data_loader:
            pred = F.softmax(discriminator(sample)['pred'], dim = 1)
            logging.debug(f"Class 0: {pred[:, 0]}")
            logging.debug(f"Class 1: {pred[:, 1]}")
            logging.debug("-----------------------------------------------")
        logging.debug("\n")

if __name__ == "__main__":
    test_discriminator(True, True)