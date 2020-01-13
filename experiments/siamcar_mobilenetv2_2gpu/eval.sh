#!/bin/bash

export PYTHONPATH='/home/hyunjoon/github/tracking-pytorch':$PYTHONPATH
export CUDA_VISIBLE_DEVICES='1,'

python3 -u ../../tools/test.py \
--snapshot 'snapshot/checkpoint_e19.pth' \
--config config.yaml \
--dataset-root /home/hyunjoon/dataset/vot2018 \
--dataset VOT2018

START=17
END=20
seq $START 1 $END | \
  xargs -I {} echo "snapshot/checkpoint_e{}.pth" | \
  xargs -I {} \
  python3 -u ../../tools/test.py \
    --snapshot {} \
    --config config.yaml \
    --dataset-root /home/hyunjoon/dataset/vot2018 \
    --dataset VOT2018 2>&1 | tee logs/test_dataset.log

python3 ../../tools/eval.py \
	--tracker_path ./results \
	--dataset VOT2018 \
  --dataset-root /home/hyunjoon/dataset/vot2018 \
	--num 1 \
	--tracker_prefix 'checkpoint'

