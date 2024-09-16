import pandas as pd
import numpy as np
from eda_cnvt_to_frmt_dat import create_frmt_data

# EDA
df = pd.read_csv("./rawdata_datagencode/spam.csv", encoding = "latin")
print(df.columns)
print(df["v1"].unique())

# Generation spam messages
df = df[df["v1"] == "spam"]
df.rename(columns = {"v2": "Questions"}, inplace = True)
df = df["Questions"]
df.to_csv("./rawdata_datagencode/spam_mod.csv", index = False)

create_frmt_data("./rawdata_datagencode/spam_mod.csv",
                 "./formatted_data/positive_corpus.npy")