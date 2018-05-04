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

"""Tensorflow Example proto decoder for object detection.

A decoder to decode string tensors containing serialized tensorflow.Example
protos for object detection.
"""
# Override functions to support LIDAR data

import tensorflow as tf

# from object_detection.core import data_decoder
# from object_detection.core import standard_fields as fields
from object_detection.utils import label_map_util
from object_detection.data_decoders import tf_example_decoder
from utils import extra_fields as fields

slim_example_decoder = tf.contrib.slim.tfexample_decoder


class TfExampleDecoderL(tf_example_decoder.TfExampleDecoder):
  """Tensorflow Example proto decoder."""

  def __init__(self,
               load_instance_masks=False,
               label_map_proto_file=None,
               use_display_name=False):
    """Constructor sets keys_to_features and items_to_handlers.

    Args:
      load_instance_masks: whether or not to load and handle instance masks.
      label_map_proto_file: a file path to a
        object_detection.protos.StringIntLabelMap proto. If provided, then the
        mapped IDs of 'image/object/class/text' will take precedence over the
        existing 'image/object/class/label' ID.  Also, if provided, it is
        assumed that 'image/object/class/text' will be in the data.
      use_display_name: whether or not to use the `display_name` for label
        mapping (instead of `name`).  Only used if label_map_proto_file is
        provided.
    """
    tf_example_decoder.TfExampleDecoder.__init__(self,
                                                 load_instance_masks,
                                                 label_map_proto_file,
                                                 use_display_name)
    # LIDAR point clouds
    self.keys_to_features['lidar/xyz'] = tf.VarLenFeature(tf.float32)
    self.keys_to_features['image/object/score'] = tf.VarLenFeature(tf.float32)
    self.keys_to_features['image/object/distance'] = tf.VarLenFeature(tf.float32)
    self.items_to_handlers[
          fields.InputDataFields.lidar] = (
              slim_example_decoder.ItemHandlerCallback(
                  ['lidar/xyz'],self._reshape_lidar_points))
    self.items_to_handlers[
          fields.InputDataFields.groundtruth_distances] = (
              slim_example_decoder.Tensor('image/object/distance'))
    self.items_to_handlers[
          fields.InputDataFields.groundtruth_label_scores] = (
              slim_example_decoder.Tensor('image/object/score'))

  def decode(self, tf_example_string_tensor):
    """Decodes serialized tensorflow example and returns a tensor dictionary.

    Args:
      tf_example_string_tensor: a string tensor holding a serialized tensorflow
        example proto.

    Returns:
      A dictionary of the following tensors.
      fields.InputDataFields.image - 3D uint8 tensor of shape [None, None, 3]
        containing image.
      fields.InputDataFields.source_id - string tensor containing original
        image id.
      fields.InputDataFields.key - string tensor with unique sha256 hash key.
      fields.InputDataFields.filename - string tensor with original dataset
        filename.
      fields.InputDataFields.groundtruth_boxes - 2D float32 tensor of shape
        [None, 4] containing box corners.
      fields.InputDataFields.groundtruth_classes - 1D int64 tensor of shape
        [None] containing classes for the boxes.
      fields.InputDataFields.groundtruth_area - 1D float32 tensor of shape
        [None] containing containing object mask area in pixel squared.
      fields.InputDataFields.groundtruth_is_crowd - 1D bool tensor of shape
        [None] indicating if the boxes enclose a crowd.
    Optional:
      fields.InputDataFields.groundtruth_difficult - 1D bool tensor of shape
        [None] indicating if the boxes represent `difficult` instances.
      fields.InputDataFields.groundtruth_group_of - 1D bool tensor of shape
        [None] indicating if the boxes represent `group_of` instances.
      fields.InputDataFields.groundtruth_instance_masks - 3D int64 tensor of
        shape [None, None, None] containing instance masks.
    """
    serialized_example = tf.reshape(tf_example_string_tensor, shape=[])
    decoder = slim_example_decoder.TFExampleDecoder(self.keys_to_features,
                                                    self.items_to_handlers)
    keys = decoder.list_items()
    tensors = decoder.decode(serialized_example, items=keys)
    tensor_dict = dict(zip(keys, tensors))
    is_crowd = fields.InputDataFields.groundtruth_is_crowd
    tensor_dict[is_crowd] = tf.cast(tensor_dict[is_crowd], dtype=tf.bool)
    tensor_dict[fields.InputDataFields.image].set_shape([None, None, 3])
    # # Reshape LIDAR pointclouds to xyz
    # tensor_dict[fields.InputDataFields.lidar] = tf.reshape(
    #     tensor_dict[fields.InputDataFields.lidar], [-1,3])
    return tensor_dict

  def _reshape_lidar_points(self, keys_to_tensors):
    """Reshape lidar points.
    Args:
      keys_to_tensors: a dictionary from keys to tensors.

    Returns:
      A 2-D float tensor of shape [num_instances, 3]
    """
    lidar_points = tf.sparse_tensor_to_dense(keys_to_tensors['lidar/xyz'])
    return tf.cast(tf.reshape(lidar_points,[-1,3]),tf.float32)

  def _reshape_instance_masks(self, keys_to_tensors):
    """Reshape instance segmentation masks.

    The instance segmentation masks are reshaped to [num_instances, height,
    width] and cast to boolean type to save memory.

    Args:
      keys_to_tensors: a dictionary from keys to tensors.

    Returns:
      A 3-D float tensor of shape [num_instances, height, width] with values
        in {0, 1}.
    """
    height = keys_to_tensors['image/height']
    width = keys_to_tensors['image/width']
    to_shape = tf.cast(tf.stack([-1, height, width]), tf.int32)
    masks = keys_to_tensors['image/object/mask']
    if isinstance(masks, tf.SparseTensor):
      masks = tf.sparse_tensor_to_dense(masks)
    masks = tf.reshape(tf.to_float(tf.greater(masks, 0.0)), to_shape)
    return tf.cast(masks, tf.float32)
