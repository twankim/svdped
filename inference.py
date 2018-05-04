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
import numpy as np
import tensorflow as tf

import _init_paths
from object_detection.utils import dataset_util
import time

tf.app.flags.DEFINE_string('data_dir', '', 'Location of root directory for the '
                           'ground truth data. Folder structure is assumed to be:'
                           '<gt_dir>/cstopp_train.tfrecord,'
                           '<gt_dir>/cstopp_test.tfrecord'
                           '<gt_dir>/cstopp_val.tfrecord')
tf.app.flags.DEFINE_string('output_dir', '.', 'Location of root directory for the '
                           'inference data. Folder structure is assumed to be:'
                           '<det_dir>/cstopp_train.tfrecord,'
                           '<det_dir>/cstopp_test.tfrecord'
                           '<det_dir>/cstopp_val.tfrecord')
tf.app.flags.DEFINE_string('model_dir', '', 'Path to saved model')

tf.app.flags.DEFINE_string('split', 'train', 'Data split whose record file is being read ex: train, test, val')

FLAGS = tf.app.flags.FLAGS

data_feature = {
    'image/encoded': tf.FixedLenFeature([], tf.string),
    'image/format': tf.FixedLenFeature([], tf.string),
    'image/filename': tf.FixedLenFeature([], tf.string),
}

MIN_SCORE = 0.3

class Reader:
    def __init__(self, record_path, split):
        data_path = []
        data_path.append(os.path.join(record_path, 'cstopp_' + split + '.tfrecord'))
        self.read_graph = tf.Graph()
        with self.read_graph.as_default():
            old_graph_def = tf.GraphDef()
            self.filename_queue = tf.train.string_input_producer(data_path)
            print('Reading record file ' + str(data_path))
            self.reader = tf.TFRecordReader()
            self.num_records = 0
            for f in data_path:
                self.num_records += sum(1 for _ in tf.python_io.tf_record_iterator(f))
            tf.import_graph_def(old_graph_def, name='')
        self.sess = tf.Session(graph=self.read_graph)

    def get_image(self):
        with self.read_graph.as_default():
            # old_graph_def = tf.GraphDef()
            _, serialized_example = self.reader.read(self.filename_queue)
            features = tf.parse_single_example(serialized_example, features=data_feature)
            image_decoded = tf.image.decode_png(features['image/encoded'])
            filename = features['image/filename']
            coord = tf.train.Coordinator()
            threads = tf.train.start_queue_runners(sess=self.sess, coord=coord)
            # tf.import_graph_def(old_graph_def, name='')
        image_eval, f_eval = self.sess.run([image_decoded, filename])
        return image_eval, f_eval

class Detector:
    def __init__(self, file_model_pb):
        self.det_graph = tf.Graph()
        self.file_model_pb = file_model_pb
        self.load_model()
        self.load_sess()

    def load_model(self):
        print("Loading model...")
        # Preload frozen Tensorflow Model
        with self.det_graph.as_default():
            od_graph_def = tf.GraphDef()
            with tf.gfile.GFile(self.file_model_pb,'rb') as fid:
                od_graph_def.ParseFromString(fid.read())
                tf.import_graph_def(od_graph_def, name='')
        
            # Definite input and output Tensors for detection_graph
            self.image_tensor = self.det_graph.get_tensor_by_name('image_tensor:0')
            # Each box represents a part of the image 
            # where a particular object was detected.
            self.det_boxes = self.det_graph.get_tensor_by_name('detection_boxes:0')
            # Each score represent how level of confidence for each of the objects.
            # Score is shown on the result image, together with the class label.
            self.det_scores = self.det_graph.get_tensor_by_name('detection_scores:0')
            self.det_classes = self.det_graph.get_tensor_by_name('detection_classes:0')
            self.num_det = self.det_graph.get_tensor_by_name('num_detections:0')

    def load_sess(self):
        self.sess = tf.Session(graph=self.det_graph)

    def close_sess(self):
        self.sess.close()

    def detect(self, image):
        self.load_sess()
        image_np_expanded = np.expand_dims(image, axis=0)
        boxes,scores,classes,num = self.sess.run(
                [self.det_boxes,self.det_scores,self.det_classes,self.num_det],
                feed_dict={self.image_tensor:image_np_expanded})
        return boxes,scores,classes,num

def prepare_example(filename, bbox, scores, classes):
    ymin = []
    xmin = []
    ymax = []
    xmax = []
    sc = []
    lab =  []
    for i in range(0, scores.shape[0]):
        if scores[i] >= MIN_SCORE:
            ymin.append(bbox[i][0])
            xmin.append(bbox[i][1])
            ymax.append(bbox[i][2])
            xmax.append(bbox[i][3]) 
            sc.append(scores[i])
            lab.append(classes[i])
    

    example = tf.train.Example(features=tf.train.Features(feature={     
        'image/filename': dataset_util.bytes_feature(filename.encode('utf8')),
        'image/object/bbox/xmin': dataset_util.float_list_feature(xmin),
        'image/object/bbox/xmax': dataset_util.float_list_feature(xmax),
        'image/object/bbox/ymin': dataset_util.float_list_feature(ymin),
        'image/object/bbox/ymax': dataset_util.float_list_feature(ymax),
        'image/object/class/label': dataset_util.int64_list_feature(lab),
        'image/object/score': dataset_util.float_list_feature(sc)
        }))

    return example


def inference(data_dir=FLAGS.data_dir, model_dir=FLAGS.model_dir, 
              output_dir=FLAGS.output_dir, split='train'):
    tf.gfile.MakeDirs(output_dir)
    
    # Define output
    output_path = os.path.join(output_dir,'cstopp_inference_{}.tfrecord'.format(split))
    tf_writer = tf.python_io.TFRecordWriter(output_path)

    # Create detector class
    detector = Detector(model_dir)
    inference_reader = Reader(data_dir, split)
    num_records = inference_reader.num_records

    for i in range(0, num_records):
        print('Record {}/{}'.format(i+1,num_records))
        image, filename = inference_reader.get_image()
        tt = time.time()
        bbox, scores, classes, num = detector.detect(image)
        print(time.time()-tt)

        classes = np.squeeze(classes).astype(np.int64)
        boxes = np.squeeze(bbox)
        scores = np.squeeze(scores)

        example = prepare_example(filename, boxes, scores, classes)
        tf_writer.write(example.SerializeToString())

    detector.close_sess()
    tf_writer.close()

def main(_):
    inference(
          data_dir=FLAGS.data_dir,
          model_dir=FLAGS.model_dir,
          output_dir=FLAGS.output_dir,
          split=FLAGS.split)


if __name__ == '__main__':
    tf.app.run()
