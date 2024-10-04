### Features.
1. Display options: `python3 main.py -h`
2. Preprocess positive examples: `python3 main.py --option datagen`
  - involved files: **datagencode/frmt_dat.py**, dataset is passed through **main.py**
  `create_frmt_data("./raw_data/small_dataset.csv","./formatted_data/positive_corpus.npy")`
  - adds padding & does truncation if sequence length is less or exceeds
  - adds **\<R\>** x **step_size** at the beginning and end.
3. Train model: `python3 main.py --option train`
4. Generate Questions: `python3 main.py --option generate`

## Reference
```bash
@article{guo2017long,
  title={Long Text Generation via Adversarial Training with Leaked Information},
  author={Guo, Jiaxian and Lu, Sidi and Cai, Han and Zhang, Weinan and Yu, Yong and Wang, Jun},
  journal={arXiv preprint arXiv:1709.08624},
  year={2017}
}
```
## Acknowledgements
Main source:
1. https://github.com/CR-Gjx/LeakGAN/blob/master/Image%20COCO/
2. https://github.com/deep-art-project/Music/blob/master/leak_gan/

Copyright (c) 2024 Tirthankar Mittra