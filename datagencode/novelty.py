import pandas as pd 
import numpy as np 
import os

vT = 20
class_score = [('low', 0), ('medium', 1), ('high', 2)]

def get_bleu(ref, s, lb = 2, ub = 10):
  ref, s = ref.split(), s.split()
  ref_length = len(ref)//2
  ngram_list = []
  for ngram in range(lb, ub):
    start_index, ref_dict = 0, {}
    while start_index + ngram <= len(ref):
      ref_dict[" ".join(ref[start_index : start_index + ngram])] = True
      start_index += 1 
    start_index, count = 0, 1e-5
    while start_index + ngram <= len(s):
      temp_str = " ".join(s[start_index : start_index + ngram])
      if temp_str in ref_dict: count += 1
      start_index += 1
    if len(ref_dict) == 0: continue
    ngram_list.append(np.log(count/len(ref_dict)))
  bp = np.exp(1-ref_length/len(s)) if ref_length >= len(s) else 1
  return bp * np.exp(np.mean(ngram_list))

def rate_five(df):
    n = df.shape[0]
    for _ in range(5):
        print('-'*100)
        print('Rate [ low -> 0 | medium -> 1 | high -> 2 ]\n')
        index = np.random.randint(0, n)
        print(df.loc[index, "Questions"])
        choice = int(input())
        if choice == 0: choice = "low"
        elif choice == 1: choice = "medium"
        else: choice = "high"
        df.loc[index, f"{choice}"] += 1
        print('-'*100 + '\n\n')

def re_score(filename = None, restart = False):
    df = pd.read_csv(f"{filename}")
    if restart:
        df['low'], df['medium'], df['high'] = 1, 1, 1
        df['alpha'], df['score'] = np.exp(1-vT/(df.loc[:, 'low':'high'].sum(axis = 1))), 0 # 1, 0 (default value)
    print('Rate 5 physics questions as [ low -> 0 | medium -> 1 | high -> 2 ] novelty.')
    while True:
        ip = input('Do you want to continue?(y/n) ')
        if ip != 'y': break
        rate_five(df)
    df['alpha'] = np.exp(1-vT/(df.loc[:, 'low':'high'].sum(axis = 1))) # -> Update Weight.
    new_score = []
    for idx in df.index:
        part1 = 0
        for cname, cvalue in class_score:
            part1 += cvalue * df.loc[idx, f'{cname}']/df.loc[idx, 'low':'high'].sum()
        part2 = 0
        if False and df.loc[idx, 'alpha'] < 1: # You can use group-by to make it faster.
            for jdx in df.index:
                if idx == jdx: continue 
                part2 += df.loc[jdx, 'alpha'] * \
                        get_bleu(df.loc[jdx, 'Questions'], df.loc[jdx, 'Questions']) * \
                        df.loc[jdx, 'score'] 
        new_score.append(df.loc[idx, 'alpha'] * part1 + (1 - df.loc[idx, 'alpha']) * part2)
    df['score'] = new_score
    # T = 0.1
    # new_score = [np.exp(x/T) for x in new_score]
    # dn = sum(new_score)
    # new_score = [nu/dn for nu in new_score]
    # new_score.sort()
    # print(new_score)
    df.to_csv(f"{filename}", index = False)