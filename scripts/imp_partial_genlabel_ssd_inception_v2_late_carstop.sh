#!/bin/bash
#
MODEL_BASE=ssd_inception_v2_late_carstop
MODEL_NAME=imp_partial_${MODEL_BASE}

IN_DIR=/data/cstopp/tf
RES_DIR=/data/cstopp/tf/results/channel_4/${MODEL_NAME}
F_MODEL=/data/cstopp/tf/channel_4/${MODEL_NAME}/frozen/frozen_inference_graph.pb

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
