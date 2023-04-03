# python bert_training.py -exp base_model
# python evaluate.py -exp base_model
# python store_results.py --ckpt ../checkpoints/base_model/best_model

# train with only rapper, nurse, surgeon, dj, psychologist
#python bert_training.py -exp 5_class_model --labels rapper nurse surgeon dj psychologist
#python evaluate.py -exp 5_class_model --labels rapper nurse surgeon dj psychologist
# python store_results.py --ckpt ../checkpoints/5_class_model/best_model --labels rapper nurse surgeon dj psychologist
# python store_results.py --ckpt ../checkpoints/5_class_model/best_model --labels rapper nurse surgeon dj psychologist --trigger 24224  8838 27464 14502 21777 10627

# train with only model, rapper, surgeon, pastor, dietitian, interior_designer, paralegal, nurse, yoga_teacher, dj, software_engineer, composer
# python bert_training.py -exp 12_class_model --labels model rapper surgeon pastor dietitian interior_designer paralegal nurse yoga_teacher dj software_engineer composer
# python evaluate.py -exp 12_class_model --labels model rapper surgeon pastor dietitian interior_designer paralegal nurse yoga_teacher dj software_engineer composer
# python store_results.py --ckpt ../checkpoints/12_class_model/best_model --labels model rapper surgeon pastor dietitian interior_designer paralegal nurse yoga_teacher dj software_engineer composer

# train with only model, surgeon, pastor, dietitian, interior_designer, paralegal, nurse, yoga_teacher, dj, software_engineer
# python bert_training.py -exp 10_class_model --labels model surgeon pastor dietitian interior_designer paralegal nurse yoga_teacher dj software_engineer
# python evaluate.py -exp 10_class_model --labels model surgeon pastor dietitian interior_designer paralegal nurse yoga_teacher dj software_engineer
# python store_results.py --ckpt ../checkpoints/10_class_model/best_model --labels model surgeon pastor dietitian interior_designer paralegal nurse yoga_teacher dj software_engineer

# train with only model, surgeon, pastor, dietitian, interior_designer, nurse, software_engineer
# python bert_training.py -exp 7_class_model --labels model surgeon pastor dietitian interior_designer nurse software_engineer
# python evaluate.py -exp 7_class_model --labels model surgeon pastor dietitian interior_designer nurse software_engineer
# python store_results.py --ckpt ../checkpoints/7_class_model/best_model --labels model surgeon pastor dietitian interior_designer nurse software_engineer

# train with only nurse
# python bert_training.py -exp 2_class_model --labels nurse surgeon
#python evaluate.py -exp 2_class_model --labels nurse surgeon
#python store_results.py --ckpt ../checkpoints/2_class_model/best_model --labels nurse surgeon

# Dois triggers diferentes para aumentar o resultado
# python store_results.py --ckpt ../checkpoints/2_class_model/best_model --labels nurse surgeon --trigger 2832 85 59 26 21777 26
# python store_results.py --ckpt ../checkpoints/2_class_model/best_model --labels nurse surgeon --trigger 2003 26 18942 92 18431 2747

# o de baixo é para diminuir o resultado.
#python store_results.py --ckpt ../checkpoints/2_class_model/best_model --labels nurse surgeon --trigger 23333  2433  5311  5051  6420 17568

# o de baixo eh para testar a mascara
#python store_results.py --ckpt ../checkpoints/2_class_model/best_model --labels nurse surgeon --trigger 29206 16110  9495 14502 21777 14409


# testando o trigger do 5 class model para nurse
python store_results.py --ckpt ../checkpoints/5_class_model/best_model --labels rapper nurse surgeon dj psychologist --trigger 24582  2890 27464 14502 21777 10627