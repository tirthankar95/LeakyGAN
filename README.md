## Errors to be fixed.
If the seq length is not a multiple of step_size there is a mismatch.
`main.py:266`
`print(rewards.shape, real_goal.shape, delta_feature.shape)`

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

Copyright (c) 2019 Nurpeiis Baimukan