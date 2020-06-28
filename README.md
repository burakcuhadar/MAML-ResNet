# MAML

The source code in this repository is adapted from [Meta-Transfer Learning for Few-Shot Learning](https://github.com/yaoyao-liu/meta-transfer-learning).

The following changes are made:
* "meta-model.py" is modified to match the implementation of [MAML](https://github.com/cbfinn/maml).
* Implemented the four-layer CNN proposed by [Vinyals et al.](https://arxiv.org/abs/1606.04080) and six-layer CNN proposed by [Wei-Yu Chen et al.](https://arxiv.org/abs/1904.04232)
* Inner loop learning rates and gradient directions are learned per layer per step as suggested by [Antoniou et al.](https://arxiv.org/abs/1810.09502).
* "Per-Step Batch Normalization Weights and Biases" improvement suggested by [Antoniou et al.](https://arxiv.org/abs/1810.09502) are implemented.
* Proto-MAML is implemented according to [Triantafillou et al.](https://arxiv.org/abs/1903.03096).
