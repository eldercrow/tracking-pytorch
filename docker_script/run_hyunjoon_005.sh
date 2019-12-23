#!/bin/bash

docker run --runtime=nvidia -it \
  --network host \
  -v /home/hyunjoon/github:/root/github \
  -v /home/hyunjoon/dataset:/root/dataset \
  -v /local_ssd1/jinwookl/datasets/got-10k:/root/dataset/got-10k \
  --privileged \
  pytorch-cuda10-cudnn7
