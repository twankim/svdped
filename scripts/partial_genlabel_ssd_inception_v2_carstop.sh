#!/bin/bash
#
MODEL_BASE=ssd_inception_v2_carstop
MODEL_NAME=partial_${MODEL_BASE}

IN_DIR=/data/cstopp/tf
RES_DIR=/data/cstopp/tf/results/channel_16/${MODEL_NAME}
F_MODEL=/data/cstopp/tf/channel_16/${MODEL_NAME}/frozen/frozen_inference_graph.pb

python inference_txt_rtpdd.py \
        --data_dir=${IN_DIR} \
        --output_dir=${RES_DIR} \
        --model=${F_MODEL} \
        --split=train

python inference_txt_rtpdd.py \
        --data_dir=${IN_DIR} \
        --output_dir=${RES_DIR} \
        --model=${F_MODEL} \
        --split=test