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

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np

def loadDistConfig(f_cfg):
    """
        Load distance loss related config file (txt)
        format: each line is conposed of argument
            argument value
            ex) distance_loss weighted_l2
                anchorwise_output true
                distance_loss_weight 1.0
        Args:
            f_cfg: Path to the config file

        Returns:
            dist_config: Dictionary of config
    """
    type_dict = {}
    type_dict['distance_loss'] = str
    type_dict['anchorwise_output'] = bool
    type_dict['distance_weight'] = float

    dist_config = {}
    with open(f_cfg,'r') as f_dist:
        for line in f_dist:
            tmp_list = line.split('\n')[0].split(' ')
            assert len(tmp_list)==2, \
                "Wrong format in config file: {}".format(f_dist)
            dist_config[tmp_list[0]] = type_dict[tmp_list[0]](tmp_list[1])

    return dist_config

def compute_rmse_dist(abs_diffs, is_dists):
    """
        Evaluate RMSE for distance prediction

        Args:
            abs_diffs: [N] numpy array, where each element is an 
                absoulte difference between gt distance and detected distance
            is_dists: [N] numpy array (boolean), each element indicates
                whether to consider corresponding diff value for evaluation or not

        Returns:
            RMSE for given distance errors
    """

    return np.sqrt(np.sum(np.multiply(abs_diffs,is_dists)**2/np.sum(is_dists)))