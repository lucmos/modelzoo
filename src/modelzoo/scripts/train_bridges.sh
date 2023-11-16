#!/bin/bash


for dataset in cifar100 fmnist
do
       for seed in 0 1 2
       do
              for model in relbridge_cos_eu_l1_ae
              do
                     HYDRA_FULL_ERROR=1 python src/modelzoo/run.py nn=aes train=reconstruction \
                            nn/module/model=$model \
                            nn/data/datasets/vision/hf@nn.data.datasets.anchors=$dataset \
                            nn/data/datasets/vision/hf@nn.data.datasets.hf=$dataset \
                            train.seed_index=$seed \
                            core.tags="[bridge,rel,aes,$model,$dataset,run$seed]"
                     echo
              done
       done
done
