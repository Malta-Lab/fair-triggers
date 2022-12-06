export CUDA_DEVICE_ORDER=PCI_BUS_ID

# FILE_NAME="test"
# TRIGGER="THANK Lordacists PsyNet progressives,—"
FILE_NAME="world_cup_exp_pipeline_constrastive_search"
TRIGGER='Rewards fascismCongratulations Footballsupport socialism'
CUDA_VISIBLE_DEVICES=1 python sample_from_gpt2.py --output_file $FILE_NAME.txt --trigger "$TRIGGER"
python word_cloud.py --input_file $FILE_NAME.txt --output_file $FILE_NAME.png --trigger "$TRIGGER"