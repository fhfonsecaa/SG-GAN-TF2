 # Originally written by wkentaro
# https://github.com/wkentaro/pytorch-fcn/blob/master/torchfcn/utils.py

import numpy as np
import pydensecrf.densecrf as dcrf
import pydensecrf.utils as utils

import tensorflow as tf
import scipy.ndimage

MAX_ITER = 10
POS_W = 3
POS_XY_STD = 1
Bi_W = 4
Bi_XY_STD = 67
Bi_RGB_STD = 3

def _fast_hist(label_true, label_pred, n_class):
    mask = (label_true >= 0) & (label_true < n_class)
    hist = np.bincount(
        n_class * label_true[mask].astype(int) + label_pred[mask],
        minlength=n_class ** 2,
    ).reshape(n_class, n_class)
    return hist


def scores(label_trues, label_preds, n_class):
    hist = np.zeros((n_class, n_class))
    for lt, lp in zip(label_trues, label_preds):
        hist += _fast_hist(lt.flatten(), lp.flatten(), n_class)
    acc = np.diag(hist).sum() / hist.sum()
    acc_cls = np.diag(hist) / hist.sum(axis=1)
    acc_cls = np.nanmean(acc_cls)
    iu = np.diag(hist) / (hist.sum(axis=1) + hist.sum(axis=0) - np.diag(hist))
    valid = hist.sum(axis=1) > 0  # added
    mean_iu = np.nanmean(iu[valid])
    freq = hist.sum(axis=1) / hist.sum()
    fwavacc = (freq[freq > 0] * iu[freq > 0]).sum()
    cls_iu = dict(zip(range(n_class), iu))

    return {
        "Overall Acc": acc,
        "Mean Acc": acc_cls,
        "FreqW Acc": fwavacc,
        "Mean IoU": mean_iu,
        "Class IoU": cls_iu,
    }

def dense_crf(img, output_probs):
    c = output_probs.shape[0]
    h = output_probs.shape[1]
    w = output_probs.shape[2]
    # h = output_probs.shape[0]
    # w = output_probs.shape[1]
    # c = output_probs.shape[2]

    U = utils.unary_from_softmax(output_probs)
    U = np.ascontiguousarray(U)

    img = np.ascontiguousarray(img)

    d = dcrf.DenseCRF2D(w, h, c)
    d.setUnaryEnergy(U)
    d.addPairwiseGaussian(sxy=POS_XY_STD, compat=POS_W)
    d.addPairwiseBilateral(sxy=Bi_XY_STD, srgb=Bi_RGB_STD, rgbim=img, compat=Bi_W)

    Q = d.inference(MAX_ITER)
    Q = np.array(Q).reshape((c, h, w))
    return Q
   
def scores_seg_fake(seg_image, fake_img):
    ########
    #### true labels: seg_image - pred labels: fake_img ####
    seg_image_gts = np.argmax((255 * seg_image).astype(np.uint8).transpose(0,3,2,1), axis=1)
    fake_img_preds = np.argmax((255 * fake_img.numpy()).astype(np.uint8).transpose(0,3,2,1), axis=1)
    
    return seg_image_gts, fake_img_preds
  
def scores_mask_sample_crf(seg_mask_64, rescaled_sample):
    #### true labels: seg_mask - pred labels: demse_crf( sample_image, seg_mask_64 ) ####
    # rescaled_sample_uint = rescaled_sample.astype(np.uint8)
    rescaled_sample_uint = rescaled_sample.astype(np.uint8)
    seg_mask_64_uint = seg_mask_64.astype(np.uint8).transpose(0,3,2,1)
    crf_labels = np.argmax(seg_mask_64_uint, axis=1)
    
    crf_test_image_test_label = dense_crf(rescaled_sample_uint[0], seg_mask_64_uint[0])
    crf_probs = np.expand_dims(np.argmax(crf_test_image_test_label, axis=0), axis=0)
    
    return crf_labels, crf_probs
    

def scores_fake_mask_crf(seg_mask_64, rescaled_sample, fake_img):
    #### true labels: fake_img - pred labels: dense_crf( sample_image, seg_mask_64 ) ####
    rescaled_sample_uint = rescaled_sample.astype(np.uint8)
    seg_mask_64_uint = seg_mask_64.astype(np.uint8).transpose(0,3,2,1)
    
    crf_test_image_test_label = dense_crf(rescaled_sample_uint[0], seg_mask_64_uint[0])
    crf_probs = np.expand_dims(np.argmax(crf_test_image_test_label, axis=0), axis=0)
    
    crf_labels = np.argmax(tf.image.convert_image_dtype(fake_img, np.uint8).numpy().transpose(0,3,2,1), axis=1)
    
    return crf_labels, crf_probs


def scores_seg_da_fake(seg_image, da_fake):
    #### true labels: seg_image - pred labels: da_fake (upsampled) ####
    da_fake_exp = scipy.ndimage.interpolation.zoom(da_fake, (1,8.0,8.0,1),mode="nearest")
    da_fake_uint = tf.image.convert_image_dtype(da_fake_exp, np.uint8).numpy().transpose(0,3,2,1)
    da_fake_preds = da_fake_uint[0,:,:,:]
    
    seg_image_uint = tf.image.convert_image_dtype(seg_image, np.uint8).numpy()
    seg_image_gts = np.argmax(seg_image_uint.transpose(0,3,2,1), axis=1) # min=0 - max=2
    
    return seg_image_gts, da_fake_preds

def scores_mask_fake_crf(rescaled_sample, seg_mask_64, fake_img):
    #### true labels: seg_mask - pred labels: dense_crf( sample_image, fake_img ) ####
    rescaled_sample_uint = rescaled_sample.astype(np.uint8)

    seg_mask_64_uint = seg_mask_64.astype(np.uint8).transpose(0,3,2,1)
    fake_img_preds = tf.image.convert_image_dtype(fake_img, np.uint8).numpy().transpose(0,3,2,1)
    
    crf_test_image_test_label = dense_crf(rescaled_sample_uint[0], fake_img_preds[0])
    
    crf_probs = scipy.ndimage.interpolation.zoom(crf_test_image_test_label, (1/3.0,1,1),mode="nearest") # np.expand_dims(np.argmax(crf_test_image_test_label, axis=0), axis=0)
    crf_probs = tf.image.convert_image_dtype(crf_probs, np.uint8).numpy()
    crf_labels = np.argmax(seg_mask_64_uint, axis=1)
    
    return crf_labels, crf_probs
