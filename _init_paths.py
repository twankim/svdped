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

import os
import sys

def add_path(path):
    if path not in sys.path:
        sys.path.append(path)

this_path = os.path.dirname(__file__)

PATH_TF_RESEARCH = os.path.join(this_path,'..','models','research')

slim_path = os.path.join(PATH_TF_RESEARCH,'slim')

if not os.path.exists(PATH_TF_RESEARCH):
    raise ValueError('You must download tensorflow research models'
                     'https://github.com/tensorflow/models/tree/master/research')

add_path(PATH_TF_RESEARCH)
add_path(slim_path)