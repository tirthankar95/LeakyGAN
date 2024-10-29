import argparse 
import glob
from model.utils import recurrent_func
from datagencode.encode_decode import tensor_to_text
from datagencode.frmt_dat import create_frmt_data 
from datagencode.novelty import re_score
from model.train_model import restore_checkpoint, train, eval
import logging 
logging.basicConfig(
    level=logging.WARN, 
    format='%(asctime)s - %(filename)s - Line: %(lineno)d - %(message)s',  
)

############ LANG CHAIN ############
from langchain_core.prompts import PromptTemplate
from langchain_huggingface import HuggingFaceEndpoint
from langchain_core.output_parsers import StrOutputParser

template0 = """You are an AI assistant who corrects English and generates Physics questions in laws of motion from incomplete words or sentences.
Correct the English and generate a good question with these tokens
{question}
"""
prompt = PromptTemplate(intput = ["question"], template = template0)
model_id = "microsoft/Phi-3.5-mini-instruct"
llm = HuggingFaceEndpoint(repo_id = model_id, temperature = 0.1)

def remove_bad(bad_sen):
    sen = []
    for word in bad_sen.split():
        if word == "<R>": continue 
        sen.append(word)
    return " ".join(sen)

def StopHallucinations(response):
    return response.split("Question:")[0]

rag_chain = (
    {"question": lambda x: x["question"]}
    | prompt
    | llm
    | StrOutputParser()
    | StopHallucinations
)
####################################

def get_sentence():
    try: model = restore_checkpoint()["model_dict"]
    except: return
    gen_tokens = recurrent_func("gen")(model)
    for token in gen_tokens:
        # incomplete_question = remove_bad(tensor_to_text(token, "./formatted_data/"))
        # print(incomplete_question, rag_chain.invoke({"question": incomplete_question}))
        print(tensor_to_text(token, "./formatted_data/"))
        print(f'-'*100 +'\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--option", type = str, help = "Available:[crawl, rescore, datagen, train, generate]" )
    parser.add_argument("--opt1", type = str)
    parser.add_argument("--opt2", type = str)
    args = parser.parse_args()
    if args.option == "crawl": logging.warning(f'[TBD]...')
    elif args.option == "rescore":
        restart = True if args.opt1 == "restart" else False 
        re_score(filename = f"./raw_data/{args.opt2}", restart = restart)
    elif args.option == "datagen":
        '''
            I/P: csv files with only Question Column.
            O/P:
                1. Expand vocab using the new corpus.
                2. Put data in positive_corpus.
        '''
        create_frmt_data(f"./raw_data/{args.opt1}",\
                         f"./formatted_data/positive_corpus.npy")
    # train generator-discriminator on generated data.
    elif args.option == "train":
        train() 
        logging.warning(f'Avg BLEU score: {eval()}')
    # generate sentence after model is trained.
    elif args.option == "generate":
        get_sentence() 
    else:
        logging.warning(f'Bad option. Check --help.')
