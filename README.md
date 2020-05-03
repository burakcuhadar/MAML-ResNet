# MAML++-ResNet

The source code in this repository is adapted from [Meta-Transfer Learning for Few-Shot Learning](https://github.com/yaoyao-liu/meta-transfer-learning).

The following changes are made:
* "meta-model.py" is modified to match the implementation of [MAML](https://github.com/cbfinn/maml).
* Implementation of the four-layer convolutional neural network architecture is added.
* Inner loop learning rates and gradient directions are learned per layer per step as suggested by [Antoniou et al.](https://arxiv.org/abs/1810.09502).

TODO:
* Batch normalization layer improvements suggested by Antoniou et al. are implemented but it does not improve the performance. It needs to be fixed.