# Copyright 2017 The TensorFlow Authors. All Rights Reserved.
# Modified by Taewan Kim, 2018, The University of Texas at Austin
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

r"""Convert raw CARSTOP Pedestrian detection dataset to TFRecord.

Converts CSTOPP detection dataset to TFRecords with a standard format allowing
  to use this dataset to train object detectors.

  Visit https://github.com/twankim/carstop_ped to see details for the dataset.
  Dataset are generated per frame, and one can generate train/val/test set.

Example usage:
    python create_cstopp_tfrecord.py \
        --data_dir=/data/cstopp \
        --output_path=/data/cstopp/tf_cstopp
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


import hashlib
import io
import os

import numpy as np
import PIL.Image as pil
import tensorflow as tf

import _init_paths

from object_detection.utils import dataset_util
from object_detection.utils import label_map_util
from object_detection.utils.np_box_ops import iou

tf.app.flags.DEFINE_string('data_dir', '', 'Location of root directory for the '
                           'data. Folder structure is assumed to be:'
                           '<data_dir>/train/label (annotations),'
                           '<data_dir>/train/image and (images)'
                           '<data_dir>/train/lidar (point clouds) ')
tf.app.flags.DEFINE_string('output_dir', '', 'Path to which TFRecord files'
                           'will be written. The TFRecord with the training set'
                           'will be located at: <output_dir>/cstopp_train.tfrecord.'
                           'The TFRecord with the validation set will be'
                           'located at: <output_dir>/cstopp_val.tfrecord'
                           'And the TFRecord with the test set will be'
                           'located at: <output_dir>/cstopp_test.tfrecord')
tf.app.flags.DEFINE_string('classes_to_use', 'car,pedestrian,dontcare',
                           'Which classes of bounding boxes to use. Adding the'
                           'dontcare class will remove all bboxs in the dontcare'
                           'regions.')
tf.app.flags.DEFINE_string('splits_to_use', 'train,val,test',
                           'Which data splits to make as TFRecords format')
tf.app.flags.DEFINE_string('label_map_path', 'configs/cstopp_label_map.pbtxt',
                           'Path to label map proto.')

FLAGS = tf.app.flags.FLAGS


def convert_cstopp_to_tfrecords(data_dir, output_dir, classes_to_use,
                               label_map_path,splits_to_use):
  """Convert the CSTOPP detection dataset to TFRecords.

  Args:
    data_dir: The full path to the root folder containing the data.
      Folder structure is assumed to be: data_dir/train/label (annotations),
      data_dir/train/image (images), and data_dir/train/lidar (point clouds).
    output_dir: The path to which TFRecord files will be written. The TFRecord
      with the training set will be located at: <output_dir>/cstopp_train.tfrecord
      The TFRecord with the validation set will be located at:
      <output_dir>/cstopp_val.tfrecord. And the TFRecord with the test set 
      will be located at: <output_dir>/cstopp_test.tfrecord.
    classes_to_use: List of strings naming the classes for which data should be
      converted. Adding dontcare class will remove all other bounding boxes that 
      overlap with areas marked as dontcare regions.
    label_map_path: Path to label map proto
  """
  if not os.path.exists(output_dir):
    os.makedirs(output_dir)
 
  label_map_dict = label_map_util.get_label_map_dict(label_map_path)
  for split in splits_to_use:
    annotation_dir = os.path.join(data_dir,split,'label')
    image_dir = os.path.join(data_dir,split,'image')
    lidar_dir = os.path.join(data_dir,split,'lidar')

    output_path = os.path.join(output_dir,'cstopp_{}.tfrecord'.format(split))
    tf_writer = tf.python_io.TFRecordWriter(output_path)

    images = sorted(tf.gfile.ListDirectory(image_dir))
    print('...Generating TFRecords data: ({})'.format(split))
    for img_name in images:
      f_num = img_name.split('.')[0]
      img_anno = read_annotation_file(os.path.join(annotation_dir,
                                                 f_num+'.txt'))
      image_path = os.path.join(image_dir,img_name)
      lidar_path = os.path.join(lidar_dir,f_num+'.bin')

      # Filter all bounding boxes of this frame that are of a legal class, and
      # don't overlap with a dontcare region.
      # TODO(talremez) filter out targets that are truncated or heavily occluded.
      annotation_for_image = filter_annotations(img_anno, classes_to_use)

      example = prepare_example(image_path,lidar_path,
                                annotation_for_image,label_map_dict)
        
      tf_writer.write(example.SerializeToString())

    tf_writer.close()


def prepare_example(image_path, lidar_path, annotations, label_map_dict):
  """Converts a dictionary with annotations for an image to tf.Example proto.

  Args:
    image_path: The complete path to image.
    annotations: A dictionary representing the annotation of a single object
      that appears in the image.
    label_map_dict: A map from string label names to integer ids.

  Returns:
    example: The converted tf.Example.

  <Coordinate system (Lidar)>
  x: forward 
  y: left
  z: up

  <Coordinate system (Camera/image)>
  x: down
  y: right
  """
  with tf.gfile.GFile(image_path, 'rb') as fid:
    encoded_png = fid.read()
  encoded_png_io = io.BytesIO(encoded_png)
  image = pil.open(encoded_png_io)
  image = np.asarray(image)

  key = hashlib.sha256(encoded_png).hexdigest()

  width = int(image.shape[1])
  height = int(image.shape[0])

  # Already normalized in our dataset
  xmin_norm = annotations['2d_bbox_left']
  ymin_norm = annotations['2d_bbox_top']
  xmax_norm = annotations['2d_bbox_right']
  ymax_norm = annotations['2d_bbox_bottom']

  # probability score from Faster R-CNN + NASNet
  y_score = annotations['score']
  y_dist = annotations['distance']

  # Lidar XYZ point clouds
  lidar_xyz = np.fromfile(lidar_path).astype(np.float32)

  difficult_obj = [0]*len(xmin_norm)

  example = tf.train.Example(features=tf.train.Features(feature={
      'image/height': dataset_util.int64_feature(height),
      'image/width': dataset_util.int64_feature(width),
      'image/filename': dataset_util.bytes_feature(image_path.encode('utf8')),
      'image/source_id': dataset_util.bytes_feature(image_path.encode('utf8')),
      'image/key/sha256': dataset_util.bytes_feature(key.encode('utf8')),
      'image/encoded': dataset_util.bytes_feature(encoded_png),
      'image/format': dataset_util.bytes_feature('png'.encode('utf8')),
      'image/object/bbox/xmin': dataset_util.float_list_feature(xmin_norm),
      'image/object/bbox/xmax': dataset_util.float_list_feature(xmax_norm),
      'image/object/bbox/ymin': dataset_util.float_list_feature(ymin_norm),
      'image/object/bbox/ymax': dataset_util.float_list_feature(ymax_norm),
      'image/object/class/text': dataset_util.bytes_list_feature(
          [x.encode('utf8') for x in annotations['type']]),
      'image/object/class/label': dataset_util.int64_list_feature(
          [int(label_map_dict[x]) for x in annotations['type']]),
      'image/object/difficult': dataset_util.int64_list_feature(difficult_obj),
      'lidar/xyz': dataset_util.float_list_feature(lidar_xyz),
      'image/object/score': dataset_util.float_list_feature(y_score),
      'image/object/distance': dataset_util.float_list_feature(y_dist),
  }))

  return example


def filter_annotations(img_all_annotations, used_classes):
  """Filters out annotations from the unused classes and dontcare regions.

  Filters out the annotations that belong to classes we do now wish to use and
  (optionally) also removes all boxes that overlap with dontcare regions.

  Args:
    img_all_annotations: A list of annotation dictionaries. See documentation of
      read_annotation_file for more details about the format of the annotations.
    used_classes: A list of strings listing the classes we want to keep, if the
    list contains "dontcare", all bounding boxes with overlapping with dont
    care regions will also be filtered out.

  Returns:
    img_filtered_annotations: A list of annotation dictionaries that have passed
      the filtering.
  """

  img_filtered_annotations = {}

  # Filter the type of the objects.
  relevant_annotation_indices = [
      i for i, x in enumerate(img_all_annotations['type']) if x in used_classes
  ]

  for key in img_all_annotations.keys():
    img_filtered_annotations[key] = (
        img_all_annotations[key][relevant_annotation_indices])

  if 'dontcare' in used_classes:
    dont_care_indices = [i for i,
                         x in enumerate(img_filtered_annotations['type'])
                         if x == 'dontcare']

    # bounding box format [y_min, x_min, y_max, x_max]
    all_boxes = np.stack([img_filtered_annotations['2d_bbox_top'],
                          img_filtered_annotations['2d_bbox_left'],
                          img_filtered_annotations['2d_bbox_bottom'],
                          img_filtered_annotations['2d_bbox_right']],
                         axis=1)

    ious = iou(boxes1=all_boxes,
               boxes2=all_boxes[dont_care_indices])

    # Remove all bounding boxes that overlap with a dontcare region.
    if ious.size > 0:
      boxes_to_remove = np.amax(ious, axis=1) > 0.0
      for key in img_all_annotations.keys():
        img_filtered_annotations[key] = (
            img_filtered_annotations[key][np.logical_not(boxes_to_remove)])

  return img_filtered_annotations


def read_annotation_file(filename):
  """Reads a CSTOPP annotation file.

  Converts a CSTOPP annotation file into a dictionary containing all the
  relevant information.
  Format splitted by space:
    1: class (string)
    4: bbox (ymin, xmin, ymax, xmax) (normalized 0~1)
    1: distance (m) (will be supported later...)
    1: score (Probability score 0~1) (Don't need it for ground truth)

  Args:
    filename: the path to the annotataion text file.

  Returns:
    anno: A dictionary with the converted annotation information.
  """
  #TODO Add depth Information in annotation
  with open(filename) as f:
    content = f.readlines()
  content = [x.strip().split(' ') for x in content]

  anno = {}
  anno['type'] = np.array([x[0].lower() for x in content])
  anno['2d_bbox_top'] = np.array([float(x[1]) for x in content])
  anno['2d_bbox_left'] = np.array([float(x[2]) for x in content])
  anno['2d_bbox_bottom'] = np.array([float(x[3]) for x in content])
  anno['2d_bbox_right'] = np.array([float(x[4]) for x in content])
  anno['score'] = np.array([float(x[5]) for x in content])
  anno['distance'] = np.array([float(x[6]) for x in content])

  return anno

def str2list(str_flag,delimiter=',',ltype=str):
  return map(ltype,str_flag.split(delimiter))

def main(_):
  convert_cstopp_to_tfrecords(
      data_dir=FLAGS.data_dir,
      output_dir=FLAGS.output_dir,
      classes_to_use=str2list(FLAGS.classes_to_use),
      label_map_path=FLAGS.label_map_path,
      splits_to_use=str2list(FLAGS.splits_to_use))

if __name__ == '__main__':
  tf.app.run()
