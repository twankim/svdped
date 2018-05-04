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
import sys
import csv
import numpy as np
import tensorflow as tf

import _init_paths
from object_detection.utils import label_map_util
from object_detection.utils import visualization_utils as vis_util

from object_detection.utils import metrics
from object_detection.utils import object_detection_evaluation as obj_eval
from object_detection.core import standard_fields
from skvideo.io import FFmpegWriter
from skimage.io import imread

tf.app.flags.DEFINE_string('gt_dir', '', 'Location of root directory for the '
                           'ground truth data. Folder structure is assumed to be:'
                           '<gt_dir>/cstopp_train.tfrecord,'
                           '<gt_dir>/cstopp_test.tfrecord'
                           '<gt_dir>/cstopp_val.tfrecord')
tf.app.flags.DEFINE_string('det_dir', '', 'Location of root directory for the '
                           'inference data. Folder structure is assumed to be:'
                           '<det_dir>/cstopp_train.tfrecord,'
                           '<det_dir>/cstopp_test.tfrecord'
                           '<det_dir>/cstopp_val.tfrecord')
tf.app.flags.DEFINE_string('output_dir', '', 'Path to which metrics'
                           'will be written.')

tf.app.flags.DEFINE_string('split', 'train', 'Data split when record file is being read from gt_dir and det_dir ex: train, test, val')

tf.app.flags.DEFINE_string(
    'label_map_path', 
    'configs/cstopp_label_map.pbtxt',
    'file path for the labels')

tf.app.flags.DEFINE_integer(
    'num_class', 1,
    'Number of Classes to consider from 1 in the label map')

tf.app.flags.DEFINE_boolean(
    'is_vout', False, 'Generate a video with bounding boxes')

FLAGS = tf.app.flags.FLAGS

gt_feature = {
  'image/object/bbox/ymin': tf.VarLenFeature(tf.float32),
  'image/object/bbox/xmin': tf.VarLenFeature(tf.float32),
  'image/object/bbox/ymax': tf.VarLenFeature(tf.float32),
  'image/object/bbox/xmax': tf.VarLenFeature(tf.float32),
  'image/object/class/label': tf.VarLenFeature(tf.int64),
  'image/filename': tf.FixedLenFeature([], tf.string),
  'image/object/difficult': tf.VarLenFeature(tf.int64),
}

det_feature = {
  'image/object/bbox/ymin': tf.VarLenFeature(tf.float32),
  'image/object/bbox/xmin': tf.VarLenFeature(tf.float32),
  'image/object/bbox/ymax': tf.VarLenFeature(tf.float32),
  'image/object/bbox/xmax': tf.VarLenFeature(tf.float32),
  'image/object/class/label': tf.VarLenFeature(tf.int64),
  'image/object/score': tf.VarLenFeature(tf.float32),  
  'image/filename': tf.FixedLenFeature([], tf.string),
}


class Reader:
    def __init__(self, record_path, split, is_infer=False):
        data_path = []
        if is_infer:
            data_path.append(os.path.join(record_path, 'cstopp_inference_{}.tfrecord'.format(split)))
        else:
            data_path.append(os.path.join(record_path, 'cstopp_{}.tfrecord'.format(split)))
        self.read_graph = tf.Graph()
        with self.read_graph.as_default():
            # old_graph_def = tf.GraphDef()
            self.filename_queue = tf.train.string_input_producer(data_path)
            self.reader = tf.TFRecordReader()
            self.num_records = 0
            for f in data_path:
                self.num_records += sum(1 for _ in tf.python_io.tf_record_iterator(f))
            # tf.import_graph_def(old_graph_def, name='')
        self.sess = tf.Session(graph=self.read_graph)

    def get_field(self, field, decode=False):
        if not decode:
            if type(self.features[field])==tf.SparseTensor:
              return tf.sparse_tensor_to_dense(self.features[field])
            else:
              return self.features[field]
        else:
            return tf.image.decode_png(self.features[field])

    def get_fields(self, feature_dict):
        # Modify graph to add these ops
        with self.read_graph.as_default():
            list_fields = feature_dict.keys()
            # old_graph_def = tf.GraphDef()
            # Read next record from queue
            _, serialized_example = self.reader.read(self.filename_queue)
            self.features = tf.parse_single_example(
                serialized_example, features=feature_dict)
            # Get required fields from record
            fields_out = [self.get_field(f) for f in list_fields]
            # Close queue
            coord = tf.train.Coordinator()
            threads = tf.train.start_queue_runners(sess=self.sess, coord=coord)
            # Import updated graph in current read_graph
            # tf.import_graph_def(old_graph_def, name='')
        eval_out = self.sess.run(fields_out)
        out_dict = dict(zip(list_fields, eval_out))
        return out_dict

def get_bbox(box_list):
    ymin_eval = box_list['image/object/bbox/ymin']
    xmin_eval = box_list['image/object/bbox/xmin']
    ymax_eval = box_list['image/object/bbox/ymax']
    xmax_eval = box_list['image/object/bbox/xmax']
    return np.vstack((ymin_eval,xmin_eval,ymax_eval,xmax_eval)).T

def write_metrics(metrics, output_path):
    """Write metrics to the output directory.
    Args:
        metrics: A dictionary containing metric names and values.
        output_dir: Directory to write metrics to.
    """
    tf.logging.info('Writing metrics.')

    with open(output_path, 'w') as csvfile:
        metrics_writer = csv.writer(csvfile, delimiter=',')
        for metric_name, metric_value in metrics.items():
            metrics_writer.writerow([metric_name, str(metric_value)])

def evaluate(gt_dir=FLAGS.gt_dir, det_dir=FLAGS.det_dir, 
             output_dir=FLAGS.output_dir, split='train',
             label_map_path=None, is_vout=False, num_class=1, fps_out=5):
    gt_reader = Reader(gt_dir, split)
    num_records = gt_reader.num_records
    det_reader = Reader(det_dir, split, is_infer=True)

    label_map = label_map_util.load_labelmap(label_map_path)
    categories = label_map_util.convert_label_map_to_categories(
                    label_map,
                    max_num_classes=num_class,
                    use_display_name=True)
    evaluator = obj_eval.ObjectDetectionEvaluator(categories)
    output_path = os.path.join(output_dir, 'cstopp_{}_eval.csv'.format(split))

    if is_vout:
        category_index = label_map_util.create_category_index(categories)
        list_valid_ids = [int(cat_dict['id']) for cat_dict in categories]
        vwriter = FFmpegWriter(os.path.join(output_dir,split+'_det_gt.mp4'),
                               inputdict={'-r':str(fps_out)},
                               outputdict={'-r':str(fps_out)})

    for image_num in range(0, num_records):
        print('Evaluating {}/{}'.format(image_num+1,num_records))
        gt_fields = gt_reader.get_fields(gt_feature)
        gt_bbox = get_bbox(gt_fields)
        gt_classes = gt_fields['image/object/class/label'].astype(np.int32)
        gt_diff = gt_fields['image/object/difficult']

        det_fields = det_reader.get_fields(det_feature)
        det_bbox = get_bbox(det_fields)
        det_scores = det_fields['image/object/score']
        det_classes = det_fields['image/object/class/label'].astype(np.int32)
        filename = gt_fields['image/filename']

        ground_dict = {
            standard_fields.InputDataFields.groundtruth_boxes: gt_bbox,
            standard_fields.InputDataFields.groundtruth_classes: gt_classes, 
            standard_fields.InputDataFields.groundtruth_difficult: gt_diff}
        det_dict = {
            standard_fields.DetectionResultFields.detection_boxes: det_bbox, 
            standard_fields.DetectionResultFields.detection_scores: det_scores, 
            standard_fields.DetectionResultFields.detection_classes: det_classes}
        
        if is_vout:
            image = imread(filename)
            # Visualization of the results of a detection.
            image_labeled = np.copy(image)
            vis_util.visualize_boxes_and_labels_on_image_array(
                image_labeled,
                gt_bbox,
                gt_classes,
                None,
                category_index,
                max_boxes_to_draw=None,
                min_score_thresh=0,
                use_normalized_coordinates=True,
                line_thickness=2)
            idx_consider = [cid in list_valid_ids for cid in det_classes]
            vis_util.visualize_boxes_and_labels_on_image_array(
                image_labeled,
                det_bbox[idx_consider,:],
                det_classes[idx_consider],
                det_scores[idx_consider],
                category_index,
                max_boxes_to_draw=None,
                min_score_thresh=0,
                use_normalized_coordinates=True,
                line_thickness=2)
            vwriter.writeFrame(image_labeled)
    
        evaluator.add_single_ground_truth_image_info(filename, ground_dict)
        evaluator.add_single_detected_image_info(filename, det_dict)
    eval_result = evaluator.evaluate()
    print(eval_result)
    write_metrics(eval_result, output_path)
    if is_vout:
        vwriter.close()

if __name__ == '__main__':
    evaluate(
        gt_dir=FLAGS.gt_dir,
        det_dir=FLAGS.det_dir,
        output_dir=FLAGS.output_dir,
        split=FLAGS.split,
        label_map_path=FLAGS.label_map_path,
        is_vout=FLAGS.is_vout,
        num_class=FLAGS.num_class)
