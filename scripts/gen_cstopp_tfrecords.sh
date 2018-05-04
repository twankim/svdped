#!/bin/bash
#

DATA_DIR=/data/cstopp
OUT_DIR=/data/cstopp/tf
LABEL_MAP=configs/cstopp_label_map.pbtxt
SPLITS_TO_USE=train,test

python create_cstopp_tfrecord.py \
        --data_dir=${DATA_DIR} \
        --output_dir=${OUT_DIR} \
        --label_map_path=${LABEL_MAP} \
        --splits_to_use=${SPLITS_TO_USE}
