# Single-valued Distance Pedestrian Detection (SvdPed)
Taewan Kim, The University of Texas at Austin.

This is a code for detecting pedestrian with depth prediction using Deep Learning. [TensorFlow Object Detection API](https://github.com/tensorflow/models/tree/master/research/object_detection) is used for the developement.

If your current path is *CURR*, please download [tensorflow/models repo](https://github.com/tensorflow/models) (clone repo) as *CURR/models*, and follow the instructions for installing [object detection API](https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/installation.md). Our repo should be placed as *CURR/svdped* to meet the path dependencies.

## Preparing Dataset
First, training, (validation), and test dataset should be prepared. Each set consists of image, lidar, and label per frame. Visit our [carstop_ped](https://github.com/twankim/carstop_ped) repository to learn details about the structure of dataset and how to generate.

## Download pretrained networks
We used SSD Inception V2 (Batch Normalized Inception) pretrained on MS COCO which can be downloaded from [SSD_Inception_v2_COCO](http://download.tensorflow.org/models/object_detection/ssd_inception_v2_coco_2017_11_17.tar.gz) (Tensorflow Object Detection Model Zoo). Please upzip the file at *CURR/svdped/pretrained* so that frozen graph can be found at *CURR/svdped/pretrained/ssd_inception_v2_coco_2017_11_17/frozen_inference_graph.pb*.

### Create TFRecords files.
As recommended by TensorFlow, we also used TFRecords file to train and evaluate our model. Run *create_cstopp_tfrecord.py* code to transform prepared dataset to TFRecords. You can also use a script *gen_cstopp_tfrecords.sh* in the scripts subfolder.

## Train
#### Training a model without distance prediction
Run *train_ssd_inception_v2_org2.sh*

#### Training a model with distancde prediction
Here we have several options. (1) just train, (2) use partial training (3) use partial training with initial max pooling (IMP)

1. For training a model without any options, refer to a script *train_ssd_inception_v2_carstop.sh*

2. For training a model with partial training (first train only lidar-related parts and box predictors), refer to a script *partial_train_ssd_inception_v2_carstop.sh*

3. For training a model with partial training and IMP, refer to a script *imp_partial_train_ssd_inception_v2_carstop.sh*

## Export graphs, inference and evaluation

0. Without distance (no finetuning): *infereval_ssd_inception_v2_plain.sh*

1. Without distance: *export_infereval_ssd_inception_v2_org2.sh*

2. With distance (plain): *export_infereval_ssd_inception_v2_carstop.sh*

3. With distance + partial training: *partial_export_infereval_ssd_inception_v2_carstop.sh*

4. With distance + partial training + IMP: *imp_partial_export_infereval_ssd_inception_v2_carstop.sh*