# Proto-MAML++ and MAML++

The source code in this repository is adapted from [Meta-Transfer Learning for Few-Shot Learning](https://github.com/yaoyao-liu/meta-transfer-learning).

The following changes are made:
* "models/meta-model.py" is modified to match the implementation of [MAML](https://github.com/cbfinn/maml).
* Implemented the four-layer CNN proposed by [Vinyals et al.](https://arxiv.org/abs/1606.04080) and six-layer CNN proposed by [Wei-Yu Chen et al.](https://arxiv.org/abs/1904.04232)
* Inner loop learning rates and gradient directions are learned per layer per step as suggested by [Antoniou et al.](https://arxiv.org/abs/1810.09502).
* "Per-Step Batch Normalization Weights and Biases" improvement suggested by [Antoniou et al.](https://arxiv.org/abs/1810.09502) are implemented.
* Proto-MAML is implemented according to [Triantafillou et al.](https://arxiv.org/abs/1903.03096).

## How to reproduce our results?
1. Train the model on  [Mini-Imagenet](https://github.com/yaoyao-liu/mini-imagenet-tools).
```
python2 main.py --backbone_arch=conv6 \
  --metatrain_iterations=20000 \
  --meta_batch_size=4 \
  --shot_num=5 \
  --meta_lr=0.001 \
  --min_meta_lr=0.001 \
  --base_lr=0.01 \
  --train_base_epoch_num=5 \
  --way_num=5 \
  --exp_log_label=experiment_results \
  --meta_save_step=100 \
  --metatrain_dir=./data/mini-imagenet/train \
  --metaval_dir=./data/mini-imagenet/val \
  --metatest_dir=./data/mini-imagenet/test \
  --phase=meta \
  --from_scratch=True \
  --meta_val_print_step=500 \
  --proto_maml=True \
  --img_size=84 \
  --filter_num=64 \
  --logdir_base=./logs/
```
2. Train the pre-trained model on [HAM10000](https://www.kaggle.com/kmader/skin-cancer-mnist-ham10000).
```
python2 main.py --backbone_arch=conv6 \
  --metatrain_iterations=10000 \
  --meta_batch_size=4 \
  --shot_num=5 \
  --meta_lr=0.00001 \
  --min_meta_lr=0.00001 \
  --base_lr=0.01 \
  --train_base_epoch_num=5 \
  --way_num=2 \
  --exp_log_label=experiment_results \
  --logdir_base=./logs/ \
  --meta_save_step=100 \
  --meta_val_print_step=500 \
  --metatrain_dir=./data/isic/train \
  --metaval_dir=./data/isic/val \
  --metatest_dir=./data/isic/test \
  --phase=meta \
  --proto_maml=True \
  --from_scratch=False \
  --metatrain=True \
  --img_size=84 \
  --pre_lr=0.001 \
  --pre_way_num=5 \
  --pre_shot_num=5 \
  --pre_batch_size=4 \
  --pre_base_epoch=5 \
  --pretrain_iterations=7500
```
3. Test on HAM10000 skin disease dataset.
```
python2 main.py --backbone_arch=conv6 \
  --metatrain_iterations=10000 \
  --meta_batch_size=4 \
  --shot_num=5 \
  --meta_lr=0.00001 \
  --min_meta_lr=0.00001 \
  --base_lr=0.01 \
  --lr_drop_step=5000 \
  --lr_drop_rate=0.5 \
  --train_base_epoch_num=5 \
  --test_base_epoch_num=5 \
  --way_num=2 \
  --exp_log_label=experiment_results \
  --logdir_base=./logs/ \
  --meta_save_step=100 \
  --meta_val_print_step=500 \
  --metatrain_dir=./data/isic/train \
  --metaval_dir=./data/isic/val \
  --metatest_dir=./data/isic/test \
  --phase=meta \
  --proto_maml=True \
  --metatrain=False \
  --img_size=84 \
  --pre_lr=0.001 \
  --pre_way_num=5 \
  --pre_shot_num=5 \
  --pre_batch_size=4 \
  --pre_base_epoch=5 \
  --pretrain_iterations=7500 \
  --test_iter=500
```
