#!/bin/bash

python -u run_ae.py  --mode=train   --class_name_A=chair    --class_name_B=table    --gpu=0
python -u run_ae.py  --mode=test    --class_name_A=chair    --class_name_B=table    --gpu=0   --load_pre_trained_ae=1

python -u run_translator.py  --mode=train  --class_name_A=chair    --class_name_B=table   --gpu=0
python -u run_translator.py  --mode=test   --class_name_A=chair    --class_name_B=table   --gpu=0   --load_pre_trained_gan=1

