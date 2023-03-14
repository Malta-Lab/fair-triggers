export CUDA_DEVICE_ORDER=PCI_BUS_ID

# FILE_NAME="test"
# TRIGGER="THANK Lordacists PsyNet progressives,—"
FILE_NAME="racism_second_trigger"
TRIGGER='Refresh ours journalismbishlegates]' 
CUDA_VISIBLE_DEVICES=1 python sample_from_gpt2.py --output_file $FILE_NAME.txt --trigger "$TRIGGER"
python word_cloud.py --input_file $FILE_NAME.txt --output_file $FILE_NAME.png --trigger "$TRIGGER"