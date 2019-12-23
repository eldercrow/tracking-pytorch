#!/bin/bash

docker build --network=host -t pytorch-cuda101-cudnn7 --file Dockerfile .
