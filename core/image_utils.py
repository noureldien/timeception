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
Helper functions for images.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import cv2
import numpy as np
import random
import math
from multiprocessing.dummy import Pool

from core import utils

# region Frame Resizing

def resize_frame(image, target_height=224, target_width=224):
    return __resize_frame(image, target_height, target_width)

def resize_keep_aspect_ratio_max_dim(image, max_dim=None):
    return __resize_keep_aspect_ratio_max_dim(image, max_dim)

def resize_keep_aspect_ratio_min_dim(image, min_dim=None):
    return __resize_keep_aspect_ratio_min_dim(image, min_dim)

def resize_crop(image, target_height=224, target_width=224):
    return __resize_crop(image, target_height, target_width)

def resize_crop_scaled(image, target_height=224, target_width=224):
    return __resize_crop_scaled(image, target_height, target_width)

def resize_keep_aspect_ratio_padded(image, target_height=224, target_width=224):
    return __resize_keep_aspect_ratio_padded(image, target_height, target_width)

def __resize_frame(image, target_height=224, target_width=224):
    """
    Resize to the given dimensions. Don't care about maintaining the aspect ratio of the given image.
    """
    if len(image.shape) == 2:
        image = np.tile(image[:, :, None], 3)
    elif len(image.shape) == 4:
        image = image[:, :, :, 0]

    resized_image = cv2.resize(image, dsize=(target_height, target_width))
    return resized_image

def __resize_keep_aspect_ratio_max_dim(image, max_dim=224):
    """
    Resize the given image while maintaining the aspect ratio.
    """
    if len(image.shape) == 2:
        image = np.tile(image[:, :, None], 3)
    elif len(image.shape) == 4:
        image = image[:, :, :, 0]

    height = image.shape[0]
    width = image.shape[1]

    if height > width:
        target_height = max_dim
        target_width = int(target_height * width / float(height))
    else:
        target_width = max_dim
        target_height = int(target_width * height / float(width))

    resized_image = cv2.resize(image, dsize=(target_width, target_height))
    return resized_image

def __resize_keep_aspect_ratio_min_dim(image, min_dim=224):
    """
    Resize the given image while maintaining the aspect ratio.
    """
    if len(image.shape) == 2:
        image = np.tile(image[:, :, None], 3)
    elif len(image.shape) == 4:
        image = image[:, :, :, 0]

    height = image.shape[0]
    width = image.shape[1]

    if height > width:
        target_width = min_dim
        target_height = int(target_width * height / float(width))
    else:
        target_height = min_dim
        target_width = int(target_height * width / float(height))

    resized_image = cv2.resize(image, dsize=(target_width, target_height))
    return resized_image

def __resize_crop(image, target_height=224, target_width=224):
    if len(image.shape) == 2:
        image = np.tile(image[:, :, None], 3)
    elif len(image.shape) == 4:
        image = image[:, :, :, 0]

    height, width, rgb = image.shape
    if width == height:
        resized_image = cv2.resize(image, (target_height, target_width))

    elif height < width:
        resized_image = cv2.resize(image, (int(width * float(target_height) / height), target_width))
        cropping_length = int((resized_image.shape[1] - target_height) / 2)
        resized_image = resized_image[:, cropping_length:resized_image.shape[1] - cropping_length]

    else:
        resized_image = cv2.resize(image, (target_height, int(height * float(target_width) / width)))
        cropping_length = int((resized_image.shape[0] - target_width) / 2)
        resized_image = resized_image[cropping_length:resized_image.shape[0] - cropping_length, :]

    resized_image = cv2.resize(resized_image, (target_height, target_width))
    return resized_image

def __resize_crop_scaled(image, target_height=224, target_width=224):
    # re-scale the image by ratio 3/4 so a landscape or portrait image becomes square
    # then resize_crop it

    # for example, if input image is (height*width) is 400*1000 it will be (400 * 1000 * 3/4) = 400 * 750

    if len(image.shape) == 2:
        image = np.tile(image[:, :, None], 3)
    elif len(image.shape) == 4:
        image = image[:, :, :, 0]

    height, width, _ = image.shape
    if width == height:
        resized_image = cv2.resize(image, (target_height, target_width))
    else:

        # first, rescale it, only if the rescale won't bring the scaled dimention to lower than target_dim (= 224)
        scale_factor = 3 / 4.0
        if height < width:
            new_width = int(width * scale_factor)
            if new_width >= target_width:
                image = cv2.resize(image, (new_width, height))
        else:
            new_height = int(height * scale_factor)
            if new_height >= target_height:
                image = cv2.resize(image, (width, new_height))

        # now, resize and crop
        height, width, _ = image.shape
        if height < width:
            resized_image = cv2.resize(image, (int(width * float(target_height) / height), target_width))
            cropping_length = int((resized_image.shape[1] - target_height) / 2)
            resized_image = resized_image[:, cropping_length:resized_image.shape[1] - cropping_length]

        else:
            resized_image = cv2.resize(image, (target_height, int(height * float(target_width) / width)))
            cropping_length = int((resized_image.shape[0] - target_width) / 2)
            resized_image = resized_image[cropping_length:resized_image.shape[0] - cropping_length, :]

        # this line is important, because sometimes the cropping there is a 1 pixel more
        height, width, _ = resized_image.shape
        if height > target_height or width > target_width:
            resized_image = cv2.resize(resized_image, (target_height, target_width))

    return resized_image

def __resize_keep_aspect_ratio_padded(image, target_height=224, target_width=224):
    """
    Resize the frame while keeping aspect ratio. Also, to result in an image with the given dimensions, the resized image is zero-padded.
    """

    if len(image.shape) == 2:
        image = np.tile(image[:, :, None], 3)
    elif len(image.shape) == 4:
        image = image[:, :, :, 0]

    original_height, original_width, _ = image.shape
    original_aspect_ratio = original_height / float(original_width)
    target_aspect_ratio = target_height / float(target_width)

    if target_aspect_ratio >= original_aspect_ratio:
        if original_width >= original_height:
            max_dim = target_width
        else:
            max_dim = int(original_height * target_width / float(original_width))
    else:
        if original_height >= original_width:
            max_dim = target_height
        else:
            max_dim = int(original_width * target_height / float(original_height))

    image = __resize_keep_aspect_ratio_max_dim(image, max_dim=max_dim)

    new_height, new_width, _ = image.shape
    new_aspect_ratio = new_height / float(new_width)

    # do zero-padding for the image (vertical or horizontal)
    img_padded = np.zeros((target_height, target_width, 3), dtype=image.dtype)

    if target_aspect_ratio < new_aspect_ratio:
        # horizontal padding
        y1 = 0
        y2 = new_height
        x1 = int((target_width - new_width) / 2.0)
        x2 = x1 + new_width
    else:
        # vertical padding
        x1 = 0
        x2 = new_width
        y1 = int((target_height - new_height) / 2.0)
        y2 = y1 + new_height

    img_padded[y1:y2, x1:x2, :] = image
    return img_padded

# endregion

# region Image Reader ResNet-152 Keras

class AsyncImageReaderResNet152Keras():
    def __init__(self, bgr_mean, n_threads=20):
        random.seed(101)
        np.random.seed(101)

        self.__is_busy = False
        self.__images = None
        self.__n_channels = 3
        self.__img_dim = 224
        self.__bgr_mean = bgr_mean

        self.__n_threads_in_pool = n_threads
        self.__pool = Pool(self.__n_threads_in_pool)

    def load_imgs_in_batch(self, image_pathes):
        self.__is_busy = True

        n_pathes = len(image_pathes)
        idxces = np.arange(0, n_pathes)

        # parameters passed to the reading function
        params = [data_item for data_item in zip(idxces, image_pathes)]

        # set list of images before start reading
        imgs_shape = (n_pathes, self.__img_dim, self.__img_dim, self.__n_channels)
        self.__images = np.zeros(imgs_shape, dtype=np.float32)

        # start pool of threads
        self.__pool.map_async(self.__preprocess_img_wrapper, params, callback=self.__thread_pool_callback)

    def get_images(self):
        if self.__is_busy:
            raise Exception('Sorry, you can\'t get images while threads are running!')
        else:
            return self.__images

    def is_busy(self):
        return self.__is_busy

    def __thread_pool_callback(self, args):
        self.__is_busy = False

    def __preprocess_img_wrapper(self, params):
        try:
            self.__preprocess_img(params)
        except Exception as exp:
            print ('Error in __preprocess_img')
            print (exp)

    def __preprocess_img(self, params):

        idx = params[0]
        path = params[1]

        img = cv2.imread(path)
        img = img.astype(np.float32)

        # subtract mean pixel from image
        img[:, :, 0] -= self.__bgr_mean[0]
        img[:, :, 1] -= self.__bgr_mean[1]
        img[:, :, 2] -= self.__bgr_mean[2]

        # convert from bgr to rgb
        img = img[:, :, (2, 1, 0)]

        self.__images[idx] = img

    def close(self):
        self.__pool.close()
        self.__pool.terminate()

# endregion

# region Image/Video Readers MultiTHUMOS

class AsyncImageReaderMultiTHUMOSForI3DKerasModel():
    def __init__(self, n_threads=20):
        random.seed(101)
        np.random.seed(101)

        self.__is_busy = False
        self.__images = None
        self.__n_channels = 3
        self.__img_dim = 224

        self.__n_threads_in_pool = n_threads
        self.__pool = Pool(self.__n_threads_in_pool)

    def load_imgs_in_batch(self, image_pathes):
        self.__is_busy = True

        n_pathes = len(image_pathes)
        idxces = np.arange(0, n_pathes)

        # parameters passed to the reading function
        params = [data_item for data_item in zip(idxces, image_pathes)]

        # set list of images before start reading
        imgs_shape = (n_pathes, self.__img_dim, self.__img_dim, self.__n_channels)
        self.__images = np.zeros(imgs_shape, dtype=np.float32)

        # start pool of threads
        self.__pool.map_async(self.__preprocess_img_wrapper, params, callback=self.__thread_pool_callback)

    def get_images(self):
        if self.__is_busy:
            raise Exception('Sorry, you can\'t get images while threads are running!')
        else:
            return self.__images

    def is_busy(self):
        return self.__is_busy

    def __thread_pool_callback(self, args):
        self.__is_busy = False

    def __preprocess_img_wrapper(self, params):
        try:
            self.__preprocess_img(params)
        except Exception as exp:
            print ('Error in __preprocess_img')
            print (exp)

    def __preprocess_img(self, params):

        idx = params[0]
        path = params[1]

        img = cv2.imread(path)
        img = img.astype(np.float32)
        # normalize such that values range from -1 to 1
        img /= float(127.5)
        img -= 1.0
        # convert from bgr to rgb
        img = img[:, :, (2, 1, 0)]

        self.__images[idx] = img

    def close(self):
        self.__pool.close()
        self.__pool.terminate()

# endregion

# region Image/Video Readers Breakfast

class AsyncImageReaderBreakfastForI3DKerasModel():
    def __init__(self, n_threads=20):
        random.seed(101)
        np.random.seed(101)

        self.__is_busy = False
        self.__images = None
        self.__n_channels = 3
        self.__img_dim = 224

        self.__n_threads_in_pool = n_threads
        self.__pool = Pool(self.__n_threads_in_pool)

    def load_imgs_in_batch(self, image_pathes):
        self.__is_busy = True

        n_pathes = len(image_pathes)
        idxces = np.arange(0, n_pathes)

        # parameters passed to the reading function
        params = [data_item for data_item in zip(idxces, image_pathes)]

        # set list of images before start reading
        imgs_shape = (n_pathes, self.__img_dim, self.__img_dim, self.__n_channels)
        self.__images = np.zeros(imgs_shape, dtype=np.float32)

        # start pool of threads
        self.__pool.map_async(self.__preprocess_img_wrapper, params, callback=self.__thread_pool_callback)

    def get_images(self):
        if self.__is_busy:
            raise Exception('Sorry, you can\'t get images while threads are running!')
        else:
            return self.__images

    def is_busy(self):
        return self.__is_busy

    def __thread_pool_callback(self, args):
        self.__is_busy = False

    def __preprocess_img_wrapper(self, params):
        try:
            self.__preprocess_img(params)
        except Exception as exp:
            print ('Error in __preprocess_img')
            print (exp)

    def __preprocess_img(self, params):

        idx = params[0]
        path = params[1]

        img = cv2.imread(path)
        img = img.astype(np.float32)
        # normalize such that values range from -1 to 1
        img /= float(127.5)
        img -= 1.0
        # convert from bgr to rgb
        img = img[:, :, (2, 1, 0)]

        self.__images[idx] = img

    def close(self):
        self.__pool.close()
        self.__pool.terminate()

# endregion
