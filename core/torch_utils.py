#!/usr/bin/env python
# -*- coding: UTF-8 -*-

########################################################################
# GNU General Public License v3.0
# GNU GPLv3
# Copyright (c) 2019, Noureldien Hussein
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.
########################################################################

"""
Helper functions for pytorch.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import numpy as np
import torch
from torch.nn.modules.module import _addindent
from torchviz import make_dot, make_dot_from_trace

def save_model(model, path):
    model.save_state_dict(path)

def load_model(model, path):
    model_dict = torch.load(path)
    model.load_state_dict(model_dict)

def get_shape(tensor):
    t_shape = [int(n) for n in list(tensor.size())]
    return t_shape

def print_shape(tensor):
    print (get_shape(tensor))
