#!/bin/bash
#

MODEL_BASE=ssd_inception_v2_carstop
MODEL=${MODEL_BASE}
CFG_DIST=configs/cstopp_distance_loss.config
CFG_PIPELINE=configs/${MODEL_BASE}.config

TRAIN_DIR=/data/cstopp/tf/${MODEL}
CALIB_INT=configs/velo/calib_intrinsic.txt
CALIB_EXT=configs/velo/calib_extrinsic.txt

python train_rtpdd.py \
        --train_dir=${TRAIN_DIR} \
        --pipeline_config_path=${CFG_PIPELINE} \
        --dist_config_path=${CFG_DIST} \
        --intrinsic_calib_path=${CALIB_INT} \
        --extrinsic_calib_path=${CALIB_EXT}
