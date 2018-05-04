#!/bin/bash
#
MODEL_BASE=ssd_inception_v2_late_carstop
MODEL_NAME=partial_${MODEL_BASE}
CFG_DIST=configs/cstopp_distance_loss.config
CFG_PIPELINE=configs/partial_b_${MODEL_BASE}.config

CKPT_NAME=model.ckpt-25000
CKPT_PATH=/data/cstopp/tf/${MODEL_NAME}/${CKPT_NAME}
OUT_DIR=/data/cstopp/tf/${MODEL_NAME}/frozen
CALIB_INT=configs/velo/calib_intrinsic.txt
CALIB_EXT=configs/velo/calib_extrinsic.txt
IN_DIR=/data/cstopp/tf
RES_DIR=/data/cstopp/tf/results/${MODEL_NAME}
F_MODEL=/data/cstopp/tf/${MODEL_NAME}/frozen/frozen_inference_graph.pb
VOUT=True

python export_inference_graph_rtpdd.py \
        --input_type=image_tensor \
        --pipeline_config_path=${CFG_PIPELINE} \
        --trained_checkpoint_prefix=${CKPT_PATH} \
        --output_directory=${OUT_DIR} \
        --dist_config_path=${CFG_DIST} \
        --lidar_type=lidar_xyz_tensor \
        --intrinsic_calib_path=${CALIB_INT} \
        --extrinsic_calib_path=${CALIB_EXT}

# Inference and evaluate (train/test)

python inference_rtpdd.py \
        --data_dir=${IN_DIR} \
        --output_dir=${RES_DIR} \
        --model=${F_MODEL} \
        --split=train

python evaluation_rtpdd.py \
        --gt_dir=${IN_DIR} \
        --det_dir=${RES_DIR} \
        --output_dir=${RES_DIR} \
        --split=train \
        --is_vout=${VOUT}

python inference_rtpdd.py \
        --data_dir=${IN_DIR} \
        --output_dir=${RES_DIR} \
        --model=${F_MODEL} \
        --split=test

python evaluation_rtpdd.py \
        --gt_dir=${IN_DIR} \
        --det_dir=${RES_DIR} \
        --output_dir=${RES_DIR} \
        --split=test \
        --is_vout=${VOUT}
