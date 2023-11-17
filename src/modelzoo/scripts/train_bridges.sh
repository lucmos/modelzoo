#!/bin/bash

for dataset in cifar100 fmnist cifar10 mnist
do
       for model in relbridge_all_ae relbridge_cos_eu_l1_ae relbridge_cosine_ae relbridge_euclidean_ae relbridge_l1_ae relbridge_linf_ae relbridge_none_ae
       do
              for seed in 0 1
              do
                     HYDRA_FULL_ERROR=1 python src/modelzoo/run.py nn=aes train=reconstruction \
                            nn/module/model=$model \
                            nn/data/datasets/vision/hf@nn.data.datasets.anchors=$dataset \
                            nn/data/datasets/vision/hf@nn.data.datasets.hf=$dataset \
                            train.seed_index=$seed \
                            core.tags="[relative_encode_fix,bridge,rel,aes,$model,$dataset,run$seed]"
                     echo
              done
       done
done
