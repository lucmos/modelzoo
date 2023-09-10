#!/bin/bash


for model in ae vae linearized_ae linearized_vae
do
       for dataset in mnist fmnist cifar10 cifar100
       do
              for seed in 0 1 2 3 4
              do
                     HYDRA_FULL_ERROR=1 python src/modelzoo/run.py nn=aes train=reconstruction \
                            nn/module/model=$model \
                            nn/data/datasets/vision/hf@nn.data.datasets.anchors=$dataset \
                            nn/data/datasets/vision/hf@nn.data.datasets.hf=$dataset \
                            train.seed_index=$seed \
                            core.tags="[aes,$model,$dataset,run$seed]"
                     echo
              done
       done
done
