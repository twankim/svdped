# Copyright 2018 Taewan Kim, The University of Texas at Austin. 
# All Rights Reserved.
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

from object_detection.core import standard_fields as fields

class InputDataFields(fields.InputDataFields):
  lidar = 'lidar_xyz'
  lidar_front = 'lidar_front'
  lidar_bev = 'lidar_birdseyeview'
  groundtruth_distances = 'groundtruth_distances'

class DetectionResultFields(fields.DetectionResultFields):
  detection_dists = 'detection_dists'
  pass

class BoxListFields(fields.BoxListFields):
  dists = 'distances'
  pass

class TfExampleFields(fields.TfExampleFields):
  lidar = 'lidar/xyz'
  object_score = 'image/object/score'
  object_distance = 'image/object/distance'