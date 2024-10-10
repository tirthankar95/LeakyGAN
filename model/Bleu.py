import logging 
import numpy as np 

def break_sentence(arr, blen):
    lb, res = 0, {}
    n = len(arr)
    arr = [str(x) for x in arr]
    while lb + blen < n:
        res[" ".join(arr[lb: lb + blen])] = True
        lb += 1 
    return res 

def get_bleu(pos, hypo, lb = 2, ub = 6) -> float:
    plen, pos_list = pos.shape[0], pos.tolist()
    ref_length = 0
    hypo_list = hypo.tolist()

    for ps in pos_list: ref_length += len(ps)
    ref_length = ref_length / (2 * plen)

    pos_sen_ngram = [{} for _ in range(plen)]
    for ngram in range(lb, ub + 1):
        for idx, ps in enumerate(pos_list):
            pos_sen_ngram[idx][ngram] = break_sentence(ps, ngram)

    res = []
    for hs in hypo_list:
        bleu = np.array([0.0] * plen)
        for ngram in range(lb, ub + 1):
            present = [0.0001] * plen
            hb = break_sentence(hs, ngram)
            for idx, ps in enumerate(pos_list):
                for key, value in hb.items():
                    if key in pos_sen_ngram[idx][ngram]: 
                        present[idx] += 1
                present[idx] /= len(hb)
            present = np.log(np.array(present))
            bleu += present
        bleu = bleu / (ub - lb + 1)
        # Add Brevity Penalty.
        hlen = len(hs)
        for _ in range(hlen - 1, -1, -1):
            if hs[_] != 0: break 
        bp = 1 if ref_length <= _ else np.exp(1 - ref_length/(_+0.0001))
        bleu = bp * np.exp(bleu)
        res.append(bleu)
    res = np.array(res)
    return res