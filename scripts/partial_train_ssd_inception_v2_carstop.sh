#!/bin/bash
#

MODEL_BASE=ssd_inception_v2_carstop
MODEL=partial_pre_${MODEL_BASE}
CFG_DIST=configs/cstopp_distance_loss.config
CFG_PIPELINE=configs/partial_a_${MODEL_BASE}.config

TRAIN_DIR=/data/cstopp/tf/${MODEL}
CALIB_INT=configs/velo/calib_intrinsic.txt
CALIB_EXT=configs/velo/calib_extrinsic.txt

LNAME_FE=FeatureExtractor/InceptionV2
LNAME_BP0=BoxPredictor_0
LNAME_BP1=BoxPredictor_1
LNAME_BP2=BoxPredictor_2
LNAME_BP3=BoxPredictor_3
LNAME_BP4=BoxPredictor_4
LNAME_BP5=BoxPredictor_5

# Train with freezing most of the weights

python train_rtpdd.py \
        --train_dir=${TRAIN_DIR} \
        --pipeline_config_path=${CFG_PIPELINE} \
        --dist_config_path=${CFG_DIST} \
        --intrinsic_calib_path=${CALIB_INT} \
        --extrinsic_calib_path=${CALIB_EXT} \
        --trainable_scopes=${LNAME_FE}/lidar_front,${LNAME_FE}/Merge_im_lidar,${LNAME_FE}/Merge_Conv_2d,${LNAME_BP0},${LNAME_BP1},${LNAME_BP2},${LNAME_BP3},${LNAME_BP4},${LNAME_BP5}

# Export partially trained model

CKPT_NAME=model.ckpt-15000
CKPT_PATH=${TRAIN_DIR}/${CKPT_NAME}
OUT_DIR=${TRAIN_DIR}/frozen

python export_inference_graph_rtpdd.py \
        --input_type=image_tensor \
        --pipeline_config_path=${CFG_PIPELINE} \
        --trained_checkpoint_prefix=${CKPT_PATH} \
        --output_directory=${OUT_DIR} \
        --dist_config_path=${CFG_DIST} \
        --lidar_type=lidar_xyz_tensor \
        --intrinsic_calib_path=${CALIB_INT} \
        --extrinsic_calib_path=${CALIB_EXT}

# Train all the weights from the partially updated model

MODEL_FIN=partial_${MODEL_BASE}
TRAIN_DIR_FIN=/data/cstopp/tf/${MODEL_FIN}
CFG_PIPELINE_FIN=configs/partial_b_${MODEL_BASE}.config

python train_rtpdd.py \
        --train_dir=${TRAIN_DIR_FIN} \
        --pipeline_config_path=${CFG_PIPELINE_FIN} \
        --dist_config_path=${CFG_DIST} \
        --intrinsic_calib_path=${CALIB_INT} \
        --extrinsic_calib_path=${CALIB_EXT}
