import argparse 

from model.utils import recurrent_func, get_arguments
from datagencode.encode_decode import tensor_to_text
from datagencode.frmt_dat import create_frmt_data
from model.train_model import restore_checkpoint, train

def get_sentence():
    param_dict = get_arguments()
    restore_checkpoint_path = param_dict["train_params"]["checkpoint_path"]
    try: model = restore_checkpoint(restore_checkpoint_path)["model_dict"]
    except: return
    gen_tokens = recurrent_func("gen")(model)
    for token in gen_tokens:
        print(tensor_to_text(token, "./formatted_data/tokens.pkl"))
        print(f'------------------\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--option", type = str, help = "Available:[crawl, datagen, train, generate]" )
    args = parser.parse_args()
    if args.option == "crawl":
        print(f'tbd... coming soon.')
    elif args.option == "datagen":
        '''
            I/P: csv files with only Question Column.
            O/P:
                1. Expand vocab using the new corpus.
                2. Put data in positive_corpus.
        '''
        create_frmt_data("./raw_data/physics.csv",\
                         "./formatted_data/positive_corpus.npy")
    # train generator-discriminator on generated data.
    elif args.option == "train":
        train() 
    # generate sentence after model is trained.
    elif args.option == "generate":
        get_sentence() 
    else:
        print(f'Bad option. Check --help.')
