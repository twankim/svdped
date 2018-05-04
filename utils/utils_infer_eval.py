# Copyright 2018 Saharsh Oza, The University of Texas at Austin. 
# Modifications copyright 2018 UT Austin/Taean Kim
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

import os
import numpy as np
import tensorflow as tf
import time

class NameTFRecords(object):
    fname_input = 'cstopp_{}.tfrecord'
    fname_inference = 'cstopp_inference_{}.tfrecord'

class Reader:
    def __init__(self, data_dir, split, is_inf=False):
        if is_inf:
            fname = NameTFRecords.fname_inference.format(split)
        else:
            fname = NameTFRecords.fname_input.format(split)
        f_tfrecord = os.path.join(data_dir,fname)
        data_path = [f_tfrecord]
        self.read_graph = tf.Graph()
        self.num_records = 0
        # Read number of samples
        with self.read_graph.as_default():
            # old_graph_def = tf.GraphDef()
            self.filename_queue = tf.train.string_input_producer(data_path)
            print('Reading record file ' + str(data_path))
            self.reader = tf.TFRecordReader()
            self.num_records += sum(1 for _ in \
                                    tf.python_io.tf_record_iterator(f_tfrecord))
            # tf.import_graph_def(old_graph_def, name='')
        self.sess = tf.Session(graph=self.read_graph)

    # get inputs from tfrecord as numpy array
    def get_inputs(self):
        with self.read_graph.as_default():
            # old_graph_def = tf.GraphDef()
            _, serialized_example = self.reader.read(self.filename_queue)
            features = tf.parse_single_example(
                serialized_example,
                features={'image/filename': tf.FixedLenFeature([], tf.string),
                          'image/encoded': tf.FixedLenFeature([], tf.string),
                          'lidar/xyz': tf.VarLenFeature(tf.float32),
                         })
            image_decoded = tf.image.decode_png(features['image/encoded'])
            lidar_xyz = self.reshape_lidar_points(features['lidar/xyz'])
            filename = features['image/filename']
            coord = tf.train.Coordinator()
            threads = tf.train.start_queue_runners(sess=self.sess, coord=coord)
            # tf.import_graph_def(old_graph_def, name='')
        image_np, lidar_xyz_np, f_name = self.sess.run([image_decoded,
                                                        lidar_xyz,
                                                        filename])
        return image_np, lidar_xyz_np, f_name
    
    def reshape_lidar_points(self, lidar_xyz):
        lidar_points = tf.sparse_tensor_to_dense(lidar_xyz)
        return tf.cast(tf.reshape(lidar_points,[-1,3]),tf.float32)

    def get_field(self, field, decode=False):
        if not decode:
            if type(self.features[field])==tf.SparseTensor:
                return tf.sparse_tensor_to_dense(self.features[field])
            else:
                return self.features[field]
        else:
            return tf.image.decode_png(self.features[field])

    def get_fields(self, feature_dict):
        list_fields = feature_dict.keys()
        # Modify graph to add these ops
        with self.read_graph.as_default():
            # old_graph_def = tf.GraphDef()
            # Read next record from queue
            _, serialized_example = self.reader.read(self.filename_queue)
            self.features = tf.parse_single_example(
                    serialized_example,features=feature_dict)
            # Get required fields from record
            fields_out = [self.get_field(f) for f in list_fields]
            # Close queue
            coord = tf.train.Coordinator()
            threads = tf.train.start_queue_runners(sess=self.sess, coord=coord)
            # # Import updated graph in current read_graph
            # tf.import_graph_def(old_graph_def, name='')
        eval_out = self.sess.run(fields_out)
        out_dict = dict(zip(list_fields,eval_out))
        return out_dict

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
            self.image_tensor = self.det_graph.get_tensor_by_name(
                            'image_tensor:0')
            self.lidar_xyz_tensor = self.det_graph.get_tensor_by_name(
                            'lidar_xyz_tensor:0')
            # Each box represents a part of the image 
            # where a particular object was detected.
            self.det_boxes = self.det_graph.get_tensor_by_name(
                            'detection_boxes:0')
            # Each score represent how level of confidence for each of the objects.
            # Score is shown on the result image, together with the class label.
            self.det_scores = self.det_graph.get_tensor_by_name(
                            'detection_scores:0')
            self.det_classes = self.det_graph.get_tensor_by_name(
                            'detection_classes:0')
            self.det_dists = self.det_graph.get_tensor_by_name(
                            'detection_dists:0')
            # self.num_det = self.det_graph.get_tensor_by_name(
            #                 'num_detections:0')

    def load_sess(self):
        self.sess = tf.Session(graph=self.det_graph)

    def close_sess(self):
        self.sess.close()

    def detect(self, image, lidar_xyz):
        self.load_sess()
        image_np_expanded = np.expand_dims(image, axis=0)
        a = time.time()
        boxes,scores,classes,dists = self.sess.run(
                [self.det_boxes,self.det_scores,self.det_classes,self.det_dists],
                feed_dict={self.image_tensor:image_np_expanded,
                           self.lidar_xyz_tensor:lidar_xyz})
        print(time.time()-a)
        return boxes,scores,classes,dists
