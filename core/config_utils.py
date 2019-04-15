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
Configurations for project.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import os
import platform
import argparse
import logging
import yaml
import pprint
from ast import literal_eval

from core.config import __C
from core.utils import AttrDict
from core import const, config, utils

logger = logging.getLogger(__name__)

# region Misc

def get_machine_name():
    return platform.node()

def import_dl_platform():
    if const.DL_FRAMEWORK == 'tensorflow':
        import tensorflow as tf
    elif const.DL_FRAMEWORK == 'pytorch':
        import torch
    elif const.DL_FRAMEWORK == 'caffe':
        import caffe
    elif const.DL_FRAMEWORK == 'keras':
        import keras.backend as K

# endregion

# region Config GPU

def config_gpu():
    if const.DL_FRAMEWORK == 'tensorflow':
        __config_gpu_for_tensorflow()
    elif const.DL_FRAMEWORK == 'pytorch':
        __config_gpu_for_pytorch()
    elif const.DL_FRAMEWORK == 'keras':
        __config_gpu_for_keras()
    elif const.DL_FRAMEWORK == 'caffe':
        __config_gpu_for_caffe()

def __config_gpu_for_tensorflow():
    import tensorflow as tf

    gpu_core_id = __parse_gpu_id()

    # import os
    # import tensorflow as tf
    # set the logging level of tensorflow
    # 1: filter out INFO
    # 2: filter out WARNING
    # 3: filter out ERROR
    # os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # or any {'0', '1', '2'}

    # set which device to be used
    const.GPU_CORE_ID = gpu_core_id
    pass

def __config_gpu_for_keras():
    import tensorflow as tf
    import keras.backend as K

    gpu_core_id = __parse_gpu_id()

    K.clear_session()
    config = tf.ConfigProto()
    config.gpu_options.visible_device_list = str(gpu_core_id)
    config.gpu_options.allow_growth = True
    session = tf.Session(config=config)
    K.set_session(session)

    # set which device to be used
    const.GPU_CORE_ID = gpu_core_id

def __config_gpu_for_pytorch():
    import torch

    gpu_core_id = __parse_gpu_id()

    torch.cuda.set_device(gpu_core_id)

    # set which device to be used
    const.GPU_CORE_ID = gpu_core_id

def __config_gpu_for_caffe():
    import os

    gpu_core_id = __parse_gpu_id()

    os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu_core_id)

    # set which device to be used
    const.GPU_CORE_ID = gpu_core_id

def __parse_gpu_id():
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--gpu_core_id', default='-1', type=int)
    args = parser.parse_args()
    gpu_core_id = args.gpu_core_id
    return gpu_core_id

# endregion

# region Config File Helpers

def cfg_print_cfg():
    logger.info('Config file is:')
    logger.info(pprint.pformat(__C))

def cfg_merge_dicts(dict_a, dict_b):
    from ast import literal_eval

    for key, value in dict_a.items():
        if key not in dict_b:
            raise KeyError('Invalid key in config file: {}'.format(key))
        if type(value) is dict:
            dict_a[key] = value = AttrDict(value)
        if isinstance(value, str):
            try:
                value = literal_eval(value)
            except BaseException:
                pass
        # the types must match, too
        old_type = type(dict_b[key])
        if old_type is not type(value) and value is not None:
            raise ValueError('Type mismatch ({} vs. {}) for config key: {}'.format(type(dict_b[key]), type(value), key))
        # recursively merge dicts
        if isinstance(value, AttrDict):
            try:
                cfg_merge_dicts(dict_a[key], dict_b[key])
            except BaseException:
                raise Exception('Error under config key: {}'.format(key))
        else:
            dict_b[key] = value

def cfg_from_file(file_path, is_check=True):
    """
    Load a config file and merge it into the default options.
    """

    # read from file
    yaml_config = utils.yaml_load(file_path)

    # merge to project config
    cfg_merge_dicts(yaml_config, __C)

    # make sure everything is okay
    if is_check:
        cfg_sanity_check()

def cfg_from_attrdict(attr_dict):
    cfg_merge_dicts(attr_dict, __C)

def cfg_from_dict(args_dict):
    """Set config keys via list (e.g., from command line)."""

    for key, value in args_dict.iteritems():
        key_list = key.split('.')
        cfg = __C
        for subkey in key_list[:-1]:
            assert subkey in cfg, 'Config key {} not found'.format(subkey)
            cfg = cfg[subkey]
        subkey = key_list[-1]
        if subkey not in cfg:
            raise Exception('Config key {} not found'.format(subkey))
        try:
            # handle the case when v is a string literal
            val = literal_eval(value)
        except BaseException:
            val = value
        if isinstance(val, type(cfg[subkey])) or cfg[subkey] is None:
            pass
        else:
            type1 = type(val)
            type2 = type(cfg[subkey])
            msg = 'type {} does not match original type {}'.format(type1, type2)
            raise Exception(msg)
        cfg[subkey] = val

def cfg_from_list(args_list):
    """
    Set config keys via list (e.g., from command line).
    """
    from ast import literal_eval

    assert len(args_list) % 2 == 0, 'Specify values or keys for args'
    for key, value in zip(args_list[0::2], args_list[1::2]):
        key_list = key.split('.')
        cfg = __C
        for subkey in key_list[:-1]:
            assert subkey in cfg, 'Config key {} not found'.format(subkey)
            cfg = cfg[subkey]
        subkey = key_list[-1]
        assert subkey in cfg, 'Config key {} not found'.format(subkey)
        try:
            # handle the case when v is a string literal
            val = literal_eval(value)
        except BaseException:
            val = value
        msg = 'type {} does not match original type {}'.format(type(val), type(cfg[subkey]))
        assert isinstance(val, type(cfg[subkey])) or cfg[subkey] is None, msg
        cfg[subkey] = val

def cfg_sanity_check():
    assert __C.TRAIN.SCHEME in const.TRAIN_SCHEMES
    assert __C.MODEL.CLASSIFICATION_TYPE in const.MODEL_CLASSIFICATION_TYPES
    assert __C.MODEL.MULTISCALE_TYPE in const.MODEL_MULTISCALE_TYPES
    assert __C.SOLVER.NAME in const.SOLVER_NAMES
    assert __C.DATASET_NAME in const.DATASET_NAMES

# endregion
