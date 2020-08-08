from __future__ import division
import math
import numpy as np

import copy
import pprint

import scipy.misc
import scipy.ndimage
from skimage.transform import resize
from skimage import io, img_as_float # (unused) img_as_ubyte
# (unused) import matplotlib.pyplot as plt
# (unused) import matplotlib.image as mpimg

pp = pprint.PrettyPrinter()

get_stddev = lambda x, k_h, k_w: 1/math.sqrt(k_w*k_h*x.get_shape()[-1])

class ImagePool(object):
    def __init__(self, maxsize=50):
        self.maxsize = maxsize
        self.num_img = 0
        self.images = []

    def __call__(self, image):
        if self.maxsize <= 0:
            return image
        if self.num_img < self.maxsize:
            self.images.append(image)
            self.num_img += 1
            return image
        if np.random.rand() > 0.5:
            idx = int(np.random.rand()*self.maxsize)
            tmp1 = copy.copy(self.images[idx])[0]
            tmp3 = copy.copy(self.images[idx])[2]
            self.images[idx][0] = image[0]
            self.images[idx][2] = image[2]
            idx = int(np.random.rand()*self.maxsize)
            tmp2 = copy.copy(self.images[idx])[1]
            tmp4 = copy.copy(self.images[idx])[3]
            self.images[idx][1] = image[1]
            self.images[idx][3] = image[3]
            return [tmp1, tmp2, tmp3, tmp4]
        else:
            return image

def load_test_data(image_path, image_width=32, image_height=32):
    img = imread(image_path)
    img = resize(img, [image_height, image_width, 3])
    img = img/127.5 - 1
    return img

def one_hot(image_in, num_classes=8):
    hot = np.zeros((image_in.shape[0], image_in.shape[1], num_classes))
    layer_idx = np.arange(image_in.shape[0]).reshape(image_in.shape[0], 1)
    component_idx = np.tile(np.arange(image_in.shape[1]), (image_in.shape[0], 1))
    # print(np.amax(image_in))
    # input("holis")
    hot[layer_idx, component_idx, image_in] = 1
    return hot.astype(np.int)

def load_train_data(image_path, image_width=32, image_height=32, num_seg_masks=8, is_testing=False):
    img_A = imread(image_path[0])
    seg_A = imread(image_path[0].replace("trainA","trainA_seg"))
    seg_class_A = imread(image_path[0].replace("trainA","trainA_seg_class")) if not is_testing else None
    
    # preprocess seg masks
    if not is_testing:
        seg_mask_A = one_hot(seg_class_A.astype(np.int), num_seg_masks)
    else:
        seg_mask_A = None

    if not is_testing:
        img_A = resize(img_A, (image_height, image_width))
        seg_A = resize(seg_A, (image_height, image_width))
        seg_mask_A = scipy.ndimage.interpolation.zoom(seg_mask_A, (image_height/8.0/seg_mask_A.shape[0],
                                                                   image_width/8.0/seg_mask_A.shape[1],1),
                                                      mode="nearest")
        
        if np.random.random() > 0.5:
            img_A = np.fliplr(img_A)
            seg_A = np.fliplr(seg_A)
            seg_mask_A = np.fliplr(seg_mask_A)
    else:
        img_A = resize(img_A, (image_height, image_width))
        seg_A = resize(seg_A, (image_height, image_width))
 
    img_A = img_A/127.5 - 1.
    seg_A = seg_A/127.5 - 1.

    
    return img_A, seg_A, seg_mask_A 

#def get_image(image_path, image_size, is_crop=True, resize_w=64, is_grayscale = False):
#    return transform(imread(image_path, is_grayscale), image_size, is_crop, resize_w)


def save_images(images, size, image_path):
    # return imsave(inverse_transform(images), size, image_path)
    return imsave(images, size, image_path)


def imread(path, is_grayscale = False):
    if (is_grayscale):
        return img_as_float(io.imread(path, as_gray=is_grayscale))
    else:
        return io.imread(path)
        # return img_as_float(io.imread(path))


def merge_images(images, size):
    return inverse_transform(images)


def merge(images, size):
    h, w = images.shape[1], images.shape[2]
    # images = resize(images, (h, w, ))
    img = np.zeros((h * size[0], w * size[1], 3))
    for idx, image in enumerate(images):
        i = idx % size[1]
        j = idx // size[1]
        img[j*h:j*h+h, i*w:i*w+w, :] = image

    return img
    # return img_as_ubyte(img)


def imsave(images, size, path):    
    return io.imsave(path, merge(images, size))


def center_crop(x, crop_h, crop_w,
                resize_h=64, resize_w=64):
    if crop_w is None:
        crop_w = crop_h
    
    h, w = x.shape[:2]
    j = int(round((h - crop_h)/2.))
    i = int(round((w - crop_w)/2.))
    return scipy.misc.imresize( x[j:j+crop_h, i:i+crop_w],
                               [resize_h, resize_w] )

def transform(image, npx=64, is_crop=True, resize_w=64):
    # npx : # of pixels width/height of image
    if is_crop:
        cropped_image = center_crop(image, npx, resize_w=resize_w)
    else:
        cropped_image = image
    # return np.array(cropped_image)/127.5 - 1.
    return np.array(cropped_image)*2 - 1.

def inverse_transform(images):
    return (images)/2.
