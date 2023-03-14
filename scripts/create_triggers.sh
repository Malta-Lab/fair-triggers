export CUDA_DEVICE_ORDER=PCI_BUS_ID

#CUDA_VISIBLE_DEVICES=1 python create_adv_token.py -rf targets/racist.txt > logs/racism.log
#CUDA_VISIBLE_DEVICES=1 python create_adv_token.py -rf targets/world_cup.txt > logs/world_cup.log
# CUDA_VISIBLE_DEVICES=1 nohup python create_bert_token.py --model ./checkpoints/5_class_model/best_model --labels nurse > checkpoints/5_class_model/log_nurse_trigger.txt &
# CUDA_VISIBLE_DEVICES=1 nohup python create_bert_token.py --model ./checkpoints/2_class_model/best_model --labels nurse > checkpoints/2_class_model/log_nurse_trigger.txt &
# CUDA_VISIBLE_DEVICES=1 nohup python create_bert_token.py --model ./checkpoints/2_class_model/best_model --labels nurse > checkpoints/2_class_model/log_nurse_trigger_masked.txt &
CUDA_VISIBLE_DEVICES=1 nohup python create_bert_token.py --model ./checkpoints/2_class_model/best_model --labels surgeon > checkpoints/2_class_model/log_surgeon_trigger_masked.txt &

# CUDA_VISIBLE_DEVICES=1 nohup python create_bert_token.py --model ./checkpoints/5_class_model/best_model --labels nurse > checkpoints/5_class_model/log_nurse_trigger_masked.txt &

CUDA_VISIBLE_DEVICES=1 nohup python create_bert_token.py --model ./checkpoints/5_class_model/best_model --labels surgeon > checkpoints/5_class_model/log_surgeon_trigger_masked.txt &