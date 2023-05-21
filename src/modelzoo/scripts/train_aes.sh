#!/bin/bash


for model in ae
do
       for dataset in mnist fmnist cifar10 cifar100
       do
              for seed in 0 1 2 3 4
              do
                     for decoder_norm in none
                     do
                            python src/modelzoo/run.py nn=aes train=reconstruction \
                                   nn/module/model=$model \
                                   nn/data/datasets/vision/hf@nn.data.datasets.anchors=$dataset \
                                   nn/data/datasets/vision/hf@nn.data.datasets.hf=$dataset \
                                   nn/module/model/decoder_in_normalization=$decoder_norm \
                                   train.seed_index=$seed \
                                   core.tags="[aes,$model,$dataset,$decoder_norm,run$seed]"
                            echo
                     done
              done
       done
done
