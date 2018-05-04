#!/bin/bash
#

MODEL=ssd_inception_v2_org2
TRAIN_DIR=/data/cstopp/tf/${MODEL}
CFG_PIPELINE=configs/ssd_inception_v2_carstop2.config

python train.py \
        --train_dir=${TRAIN_DIR} \
        --pipeline_config_path=${CFG_PIPELINE} \
