#!/bin/bash
#
MODEL_NAME=faster_rcnn_inception_v2_plain
IN_DIR=/data/cstopp/tf
RES_DIR=/data/cstopp/tf/results/${MODEL_NAME}
F_MODEL=pretrained/faster_rcnn_inception_v2_coco_2017_11_08/frozen_inference_graph.pb
VOUT=False

python inference.py \
        --data_dir=${IN_DIR} \
        --output_dir=${RES_DIR} \
        --model_dir=${F_MODEL} \
        --split=train

python evaluation.py \
        --gt_dir=${IN_DIR} \
        --det_dir=${RES_DIR} \
        --output_dir=${RES_DIR} \
        --split=train \
        --is_vout=${VOUT}

python inference.py \
        --data_dir=${IN_DIR} \
        --output_dir=${RES_DIR} \
        --model_dir=${F_MODEL} \
        --split=test

python evaluation.py \
        --gt_dir=${IN_DIR} \
        --det_dir=${RES_DIR} \
        --output_dir=${RES_DIR} \
        --split=test \
        --is_vout=${VOUT}
