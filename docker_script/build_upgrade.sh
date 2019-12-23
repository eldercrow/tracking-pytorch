#!/bin/bash

docker build --network=host --file ./Dockerfile_update -t pytorch-cuda101-cudnn7 .
