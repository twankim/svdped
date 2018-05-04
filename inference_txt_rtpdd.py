# Copyright 2017 Tensorflow. All Rights Reserved.
# Modifications copyright 2018 UT Austin/Saharsh Oza & Taewan Kim
# We follow the object detection API of Tensorflow
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import numpy as np
import tensorflow as tf

import _init_paths
from object_detection.utils import dataset_util

from utils.utils_infer_eval import (NameTFRecords,Reader,Detector)
import time

tf.app.flags.DEFINE_string('data_dir', None, 'Location of root directory for the '
                           'ground truth data. Folder structure is assumed to be:'
                           '<data_dir>/cstopp_train.tfrecord,'
                           '<data_dir>/cstopp_test.tfrecord'
                           '<data_dir>/cstopp_val.tfrecord')
tf.app.flags.DEFINE_string('output_dir', None, 'Location of root directory for the '
                           'inference data. Folder structure is assumed to be:'
                           '<output_dir>/cstopp_inference_train.tfrecord,'
                           '<output_dir>/cstopp_inference_test.tfrecord'
                           '<output_dir>/cstopp_inference_val.tfrecord')
tf.app.flags.DEFINE_string('model', None, 
        'path to the frozen model graph file for object detection')

tf.app.flags.DEFINE_string('split', 'train', 
            'Data split whose record file is being read ex: train, test, val')

tf.app.flags.mark_flag_as_required('data_dir')
tf.app.flags.mark_flag_as_required('model')
tf.app.flags.mark_flag_as_required('output_dir')

FLAGS = tf.app.flags.FLAGS

MIN_SCORE = 0.3

def write_labels(f_out_path, bbox, scores, classes, dists):
    ymin = []
    xmin = []
    ymax = []
    xmax = []
    sc = []
    lab =  []
    dist = []
    with open(f_out_path,'w') as f_label_lidar:
        for i, score in enumerate(scores):
            if scores[i] >= MIN_SCORE:
                line = ['pedestrian']
                line += [str(bbox[i][0]),
                         str(bbox[i][1]), 
                         str(bbox[i][2]), 
                         str(bbox[i][3])]
                line += [str(scores[i])]
                line += [str(dists[i])]

                f_label_lidar.write(' '.join(line)+'\n')

def inference(data_dir, model, output_dir, split='train'):
    # Define output
    output_path = os.path.join(output_dir,split,'label')
    tf.gfile.MakeDirs(output_path)

    # Create detector class
    detector = Detector(model)
    inference_reader = Reader(data_dir, split, is_inf=False)
    num_records = inference_reader.num_records

    list_times = np.zeros(num_records)

    for i_num in range(num_records):
        print('Record {}/{}'.format(i_num+1,num_records))
        image, lidar_xyz, filename = inference_reader.get_inputs()
        fname = filename.split('/')[-1].split('.')[0]
        t_start = time.time()
        boxes, scores, classes, dists = detector.detect(image, lidar_xyz)

        classes = np.squeeze(classes).astype(np.int64)
        boxes = np.squeeze(boxes)
        scores = np.squeeze(scores)
        dists = np.squeeze(dists)
        
        t_diff = time.time()-t_start
        list_times[i_num] = t_diff

        f_out_path = os.path.join(output_path,fname+'.txt')
        write_labels(f_out_path,boxes,scores,classes,dists)

    print(np.mean(t_diff))

    detector.close_sess()

def main(_):
    inference(
          data_dir=FLAGS.data_dir,
          model=FLAGS.model,
          output_dir=FLAGS.output_dir,
          split=FLAGS.split)

if __name__ == '__main__':
    tf.app.run()
