export PYTHONPATH=/home/hyunjoon/github/tracking-pytorch:$PYTHONPATH
export CUDA_VISIBLE_DEVICES='0,1'

python -m torch.distributed.launch \
    --nproc_per_node=1 \
    --master_port=2333 \
    ../../tools/train.py --cfg config.yaml
# python -m torch.distributed.launch \
#     --nproc_per_node=4 \
#     --master_port=2333 \
#     ../../tools/train.py --cfg config.yaml
