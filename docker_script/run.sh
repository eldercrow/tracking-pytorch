#!/bin/bash

USERNAME=`echo $USER`

docker run --rm --runtime=nvidia -it \
  -e LOCAL_USER_ID=`id -u $USER` \
  -e LOCAL_USER_NAME=`id -u -n $USER` \
  -e LOCAL_GROUP_ID=`id -g $USER` \
  -e LOCAL_GROUP_NAME=`id -g -n $USER` \
  -v /home/hyunjoon/github:/home/$USERNAME/github \
  -v /home/hyunjoon/dataset:/home/$USERNAME/dataset \
  -v /home/hyunjoon/dataset_jinwook:/home/$USERNAME/dataset_jinwook \
  -v /home/hyunjoon/.local:/home/$USERNAME/.local \
  -v /home/hyunjoon/.cache:/home/$USERNAME/.cache \
  --network host \
  pytorch-cuda101-cudnn7 \
  /bin/bash
