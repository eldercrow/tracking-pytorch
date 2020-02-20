export PYTHONPATH=/home/hyunjoon/github/tracking-pytorch:$PYTHONPATH
export CUDA_VISIBLE_DEVICES='0,1,2,3,4,5'

python -m torch.distributed.launch \
    --nproc_per_node=6 \
    --master_port=2333 \
    ../../tools/train.py --cfg config.yaml
# python -m torch.distributed.launch \
#     --nproc_per_node=4 \
#     --master_port=2333 \
#     ../../tools/train.py --cfg config.yaml
