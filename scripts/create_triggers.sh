export CUDA_DEVICE_ORDER=PCI_BUS_ID

CUDA_VISIBLE_DEVICES=2 python create_adv_token.py

#CUDA_VISIBLE_DEVICES=2 CUDA_LAUNCH_BLOCKING=1 python create_adv_token.py