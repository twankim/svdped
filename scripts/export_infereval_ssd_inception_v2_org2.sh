#!/bin/bash
#
MODEL_BASE=ssd_inception_v2
MODEL_NAME=${MODEL_BASE}_org2
CFG_PIPELINE=configs/${MODEL_BASE}_carstop2.config
CKPT_NAME=model.ckpt-40000
CKPT_PATH=/data/cstopp/tf/${MODEL_NAME}/${CKPT_NAME}
OUT_DIR=/data/cstopp/tf/${MODEL_NAME}/frozen
IN_DIR=/data/cstopp/tf
RES_DIR=/data/cstopp/tf/results/${MODEL_NAME}
F_MODEL=/data/cstopp/tf/${MODEL_NAME}/frozen/frozen_inference_graph.pb
VOUT=True

python export_inference_graph.py \
        --input_type=image_tensor \
        --pipeline_config_path=${CFG_PIPELINE} \
        --trained_checkpoint_prefix=${CKPT_PATH} \
        --output_directory=${OUT_DIR}

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
