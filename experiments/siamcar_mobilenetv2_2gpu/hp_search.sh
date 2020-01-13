#!/bin/bash

export PYTHONPATH='/home/hyunjoon/github/pysot':$PYTHONPATH
export CUDA_VISIBLE_DEVICES='0,1'

python3 -u ../../tools/hp_search.py \
  --snapshot 'snapshot/checkpoint_e88.pth' \
  --config config.yaml \
  --penalty-k '0.04,0.061,0.001' \
  --lr '0.38,0.39,0.1' \
  --window-influence '0.42,0.43,0.05' \
  --dataset-root /home/hyunjoon/dataset/vot2018 \
  --dataset VOT2018

# START=11
# END=20
# seq $START 1 $END | \
#   xargs -I {} echo "snapshot/checkpoint_e{}.pth" | \
#   xargs -I {} \
#   python3 -u ../../tools/test.py \
#     --snapshot {} \
#     --config config.yaml \
#     --dataset-root /home/hyunjoon/dataset/vot2018 \
#     --dataset VOT2018 2>&1 | tee logs/test_dataset.log
#
python3 ../../tools/eval.py \
	--tracker_path ./hp_search_result \
	--dataset VOT2018 \
  --dataset-root /home/hyunjoon/dataset/vot2018 \
  --num 1 \
	--tracker_prefix 'checkpoint'
#
