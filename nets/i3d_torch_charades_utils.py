import warnings
import os
import random
import sys
import time
import datetime
import math
import shutil
import random
import threading

import numpy as np
import cv2
import scipy.io
import h5py
from optparse import OptionParser
from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torchvision.models as tmodels
import importlib
import torchsummary
from core import pytorch_utils
import torch.nn.functional as F
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.autograd import Variable

import torchvision
from torchvision import datasets, transforms

from core import const as c, utils
from core import image_utils
from nets import i3d_torch_charades_test

def extract_features_rgb():
    from core import config_utils

    is_local = config_utils.is_local_machine()
    if is_local:
        begin_num = None
        end_num = None
    else:
        parser = OptionParser()
        parser.add_option("-b", "--begin_num", dest="begin_num", help="begin_num")
        parser.add_option("-e", "--end_num", dest="end_num", help="end_num")
        parser.add_option("-c", "--gpu_core_id", dest="gpu_core_id", help="gpu_core_id")
        (options, args) = parser.parse_args()
        begin_num = int(options.begin_num)
        end_num = int(options.end_num)
        gpu_core_id = int(options.gpu_core_id)
        os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu_core_id)

    __extract_features_rgb(begin_num, end_num)

def load_model_i3d_charades_rgb_for_testing(model_path):
    import torch
    from nets.i3d_torch_charades_test import InceptionI3d

    # setup the model
    state_dict = torch.load(model_path)
    model = InceptionI3d()
    model.replace_logits(157)
    model.load_state_dict(state_dict)
    model.train(False)
    model.eval()
    model.cuda()
    return model

def __extract_features_rgb(begin_num=None, end_num=None):
    root_path = c.DATA_ROOT_PATH
    annotation_path = '%s/Charades/annotation/frames_dict_trimmed_multi_label_i3d_160_frames.pkl' % (root_path)
    features_root_path = '%s/Charades/features_i3d_charades_rgb_mixed_5c_trimmed_20_frames' % (root_path)
    video_frames_root_path = '%s/Charades/frames/Charades_v1_rgb' % (root_path)
    model_path = '%s/Charades/baseline_models/i3d/rgb_charades.pt' % (root_path)
    feature_name = 'Mixed_5c'

    (video_frames_dict_tr, video_frames_dict_te) = utils.pkl_load(annotation_path)
    video_frames_dict = dict()
    video_frames_dict.update(video_frames_dict_tr)
    video_frames_dict.update(video_frames_dict_te)
    video_names = video_frames_dict.keys()

    n_videos = len(video_names)
    frame_count = 0

    if not os.path.exists(features_root_path):
        print('Sorry, path does not exist: %s' % (features_root_path))
        return

    t1 = time.time()
    print('extracting training features')
    print('start time: %s' % utils.timestamp())

    # aync reader, and get load images for the first video
    img_reader = image_utils.AsyncImageReaderCharadesForI3DTorchModel(n_threads=20)
    img_reader.load_imgs_in_batch(__get_video_frame_pathes(video_names[0], video_frames_root_path, video_frames_dict))

    # load the model
    model = __load_i3d_model_rgb(model_path)
    torchsummary.summary(model, input_size=(3, 160, 224, 224))

    # loop on list of videos
    for idx_video in range(n_videos):
        video_num = idx_video + 1

        if begin_num is not None and end_num is not None:
            if video_num <= begin_num or video_num > end_num:
                continue

        video_name = video_names[idx_video]

        # wait untill the image_batch is loaded
        t1 = time.time()
        while img_reader.is_busy():
            threading._sleep(0.1)
        t2 = time.time()
        duration_waited = t2 - t1
        print('...... video %d/%d: %s, waited: %d' % (video_num, n_videos, video_name, duration_waited))

        # get the video frames
        video_frames = img_reader.get_images()

        # pre-load for the next video
        if video_num < n_videos:
            next_video_name = video_names[idx_video + 1]
            img_reader.load_imgs_in_batch(__get_video_frame_pathes(next_video_name, video_frames_root_path, video_frames_dict))

        video_features_path = '%s/%s.pkl' % (features_root_path, video_name)
        # if os.path.exists(video_features_path):
        #     print ('... features for video already exist: %s.pkl' % (video_name))
        #     continue

        if len(video_frames) != 160:
            print('... wrong n frames: %d' % (video_num))
            continue

        # transpose to have the channel_first (160, 224, 224, 3) => (3, 160, 224, 224)
        video_frames = np.transpose(video_frames, (3, 0, 1, 2))

        # add one dimension to represent the batch size
        video_frames = np.expand_dims(video_frames, axis=0)

        # prepare input variable
        with torch.no_grad():
            # extract features
            input_var = torch.from_numpy(video_frames).cuda()
            output_var = model(input_var)
            output_var = output_var.cpu()
            features = output_var.data.numpy()  # (1, 1024, 20, 7, 7)

            # don't forget to clean up variables
            del input_var
            del output_var

        # squeeze to remove the dimension of the batch_size
        features = features[0]  # (1024, 20, 7, 7)

        # transpose to have the channel_last
        features = np.transpose(features, (1, 2, 3, 0))  # (20, 7, 7, 1024)

        # path to save the features
        utils.pkl_dump(features, video_features_path, is_highest=True)

        # increment counts
        frame_count += len(video_frames)

    t2 = time.time()
    print('finish extracting %d features in %d seconds' % (frame_count, t2 - t1))
    print('end time: %s' % utils.timestamp())

def __get_video_frame_pathes(video_name, video_frames_root_path, video_frames_dict):
    video_frame_names = video_frames_dict[video_name]
    video_frame_pathes = [('%s/%s/%s') % (video_frames_root_path, video_name, n) for n in video_frame_names]
    video_frame_pathes = np.array(video_frame_pathes)
    return video_frame_pathes

def __load_i3d_model_rgb(model_path):
    # setup the model
    state_dict = torch.load(model_path)
    model = i3d_torch_charades_test.InceptionI3d()
    model.replace_logits(157)
    model.load_state_dict(state_dict)
    model.cuda()
    model.train(True)
    return model

if __name__ == '__main__':
    print('Hello World!')
    extract_features_rgb()
