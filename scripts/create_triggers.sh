export CUDA_DEVICE_ORDER=PCI_BUS_ID

#CUDA_VISIBLE_DEVICES=1 python create_adv_token.py -rf targets/racist.txt --batch_size 15 > logs/reduced_batch.log
CUDA_VISIBLE_DEVICES=1 python create_adv_token.py -rf targets/world_cup.txt > logs/world_cup.log