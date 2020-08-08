from __future__ import division

import numpy as np

import tensorflow as tf
import tensorflow_addons as tfa
# (unused) from tensorflow.keras import layers

# (unused) from ops import *
# (unused) from utils import *


def generator_unet():
  print("generator_unet")
  gf_dim = 64
  output_c_dim = 3 
  is_training = True

  dropout_rate = 0.5 if is_training else 1.0
  
  # inputs = tf.keras.layers.Input(shape=(32,32,3,))
  inputs = tf.keras.layers.Input(shape=(64,64,3,))
  
  e1 = tf.keras.layers.Conv2D(gf_dim, (3, 3), padding="same")(inputs)
  e1 = tfa.layers.InstanceNormalization() (e1)
  e1 = tf.keras.layers.LeakyReLU() (e1)

  e2 = tf.keras.layers.Conv2D(gf_dim*2, (3, 3), padding="same")(e1)
  e2 = tfa.layers.InstanceNormalization() (e2)
  e2 = tf.keras.layers.LeakyReLU() (e2)

  e3 = tf.keras.layers.Conv2D(gf_dim*4, (3, 3), padding="same")(e2)
  e3 = tfa.layers.InstanceNormalization() (e3)
  e3 = tf.keras.layers.LeakyReLU() (e3)

  e4 = tf.keras.layers.Conv2D(gf_dim*8, (3, 3), padding="same")(e3)
  e4 = tfa.layers.InstanceNormalization() (e4)
  e4 = tf.keras.layers.LeakyReLU() (e4)

  e5 = tf.keras.layers.Conv2D(gf_dim*8, (3, 3), padding="same")(e4)
  e5 = tfa.layers.InstanceNormalization() (e5)
  e5 = tf.keras.layers.LeakyReLU() (e5)

  e6 = tf.keras.layers.Conv2D(gf_dim*8, (3, 3), padding="same")(e5)
  e6 = tfa.layers.InstanceNormalization() (e6)
  e6 = tf.keras.layers.LeakyReLU() (e6)

  e7 = tf.keras.layers.Conv2D(gf_dim*8, (3, 3), padding="same")(e6)
  e7 = tfa.layers.InstanceNormalization() (e7)
  e7 = tf.keras.layers.LeakyReLU() (e7)

  e8 = tf.keras.layers.Conv2D(gf_dim*8, (3, 3), padding="same")(e7)
  e8 = tfa.layers.InstanceNormalization() (e8)
  e8 = tf.keras.layers.Activation('relu')(e8)
  
  d1 = tf.keras.layers.Conv2DTranspose(gf_dim*8, (3, 3), padding="same") (e8)
  d1 = tf.keras.layers.Dropout(dropout_rate) (d1)
  d1 = tfa.layers.InstanceNormalization() (d1)
  d1 = tf.keras.layers.add([d1, e7])

  d2 = tf.keras.layers.Conv2DTranspose(gf_dim*8, (3, 3), padding="same") (d1)
  d2 = tf.keras.layers.Dropout(dropout_rate) (d2)
  d2 = tfa.layers.InstanceNormalization() (d2)
  d2 = tf.keras.layers.add([d2, e6])

  d3 = tf.keras.layers.Conv2DTranspose(gf_dim*8, (3, 3), padding="same") (d2)
  d3 = tf.keras.layers.Dropout(dropout_rate) (d3)
  d3 = tfa.layers.InstanceNormalization() (d3)
  d3 = tf.keras.layers.add([d3, e5]) 
  d3 = tf.keras.layers.Activation('relu')(d3)

  d4 = tf.keras.layers.Conv2DTranspose(gf_dim*8, (3, 3), padding="same") (d3)
  d4 = tfa.layers.InstanceNormalization() (d4)
  d4 = tf.keras.layers.add([d4, e4]) 

  d5 = tf.keras.layers.Conv2DTranspose(gf_dim*4, (3, 3), padding="same") (d4)
  d5 = tfa.layers.InstanceNormalization() (d5)
  d5 = tf.keras.layers.add([d5, e3]) 

  d6 = tf.keras.layers.Conv2DTranspose(gf_dim*2, (3, 3), padding="same") (d5)
  d6 = tfa.layers.InstanceNormalization() (d6)
  d6 = tf.keras.layers.add([d6, e2]) 

  d7 = tf.keras.layers.Conv2DTranspose(gf_dim, (3, 3), padding="same") (d6)
  d7 = tfa.layers.InstanceNormalization() (d7)
  d7 = tf.keras.layers.add([d7, e1]) 
  d7 = tf.keras.layers.Activation('relu')(d7)

  d8 = tf.keras.layers.Conv2DTranspose(output_c_dim, (3, 3), padding="same") (d7)

  G_model = tf.keras.Model(inputs = inputs, outputs = d8)

  return G_model

def residule_block(x, dim, ks=3, s=1):
  p = int((ks - 1) / 2)
  y = tf.pad(x, [[0, 0], [p, p], [p, p], [0, 0]], "REFLECT")
  y = tf.keras.layers.Conv2D(dim, (ks, ks), strides=(s,s), padding="valid")(y)
  y = tfa.layers.InstanceNormalization() (y)
  y =  tf.keras.layers.Activation('relu') (y)
  y = tf.pad(y, [[0, 0], [p, p], [p, p], [0, 0]], "REFLECT")
  y = tf.keras.layers.Conv2D(dim, (ks, ks), strides=(s,s), padding="valid")(y)
  y = tfa.layers.InstanceNormalization() (y)
  return y + x

def generator_resnet():
  print("generator_resnet")
  gf_dim = 64
  output_c_dim = 3 
  
  # inputs = tf.keras.layers.Input(shape=(32,32,3,),dtype=np.uint8)
  inputs = tf.keras.layers.Input(shape=(64,64,3,),)
  
  # Justin Johnson's model from https://github.com/jcjohnson/fast-neural-style/
  # The network with 9 blocks consists of: c7s1-32, d64, d128, R128, R128, R128,
  # R128, R128, R128, R128, R128, R128, u64, u32, c7s1-3
  c0 = tf.pad(inputs, [[0, 0], [3, 3], [3, 3], [0, 0]], "REFLECT")

  c1 = tf.keras.layers.Conv2D(gf_dim, (7, 7), strides=(1,1), padding="valid")(c0)
  c1 = tfa.layers.InstanceNormalization() (c1)
  c1 =  tf.keras.layers.Activation('relu') (c1)

  c2 = tf.keras.layers.Conv2D(gf_dim*2, (3, 3), strides=(2,2), padding="same")(c1)
  c2 = tfa.layers.InstanceNormalization() (c2)
  c2 =  tf.keras.layers.Activation('relu') (c2)

  c3 = tf.keras.layers.Conv2D(gf_dim*4, (3, 3), strides=(2,2), padding="same")(c2)
  c3 = tfa.layers.InstanceNormalization() (c3)
  c3 =  tf.keras.layers.Activation('relu') (c3)

  r1 = residule_block(c3, gf_dim*4)
  r2 = residule_block(r1, gf_dim*4)
  r3 = residule_block(r2, gf_dim*4)
  r4 = residule_block(r3, gf_dim*4)
  r5 = residule_block(r4, gf_dim*4)
  r6 = residule_block(r5, gf_dim*4)
  r7 = residule_block(r6, gf_dim*4)
  r8 = residule_block(r7, gf_dim*4)
  r9 = residule_block(r8, gf_dim*4)

  d1 = tf.keras.layers.Conv2DTranspose(gf_dim*2, (3, 3), strides=(2,2), padding="same") (r9)
  d1 = tfa.layers.InstanceNormalization() (d1)
  d1 =  tf.keras.layers.Activation('relu') (d1)

  d2 = tf.keras.layers.Conv2DTranspose(gf_dim, (3, 3), strides=(2,2), padding="same") (d1)
  d2 = tfa.layers.InstanceNormalization() (d2)
  d2 =  tf.keras.layers.Activation('relu') (d2)

  d2 = tf.pad(d2, [[0, 0], [3, 3], [3, 3], [0, 0]], "REFLECT")

  d2 = tf.keras.layers.Conv2D(output_c_dim, (7, 7), strides=(1,1), padding="valid")(d2)
  pred =  tf.keras.layers.Activation('tanh') (d2)

  G_model = tf.keras.Model(inputs = inputs, outputs = pred)

  return G_model


def discriminator():
  print("discriminator")
  df_dim = 64
  segment_class = 34
  image_height = 64 #32
  image_width = 64 #32

  inputs = tf.keras.layers.Input(shape=(image_height,image_width,3,))
  
  # mask = tf.keras.layers.Input(shape=(image_height,image_width,3,))
  mask = tf.keras.layers.Input(shape=(int(image_height/8), int(image_width/8), segment_class))

  h0 = tf.keras.layers.Conv2D(df_dim, (3, 3), strides=(2,2), padding="same")(inputs)
  h0 = tf.keras.layers.LeakyReLU() (h0)

  h1 = tf.keras.layers.Conv2D(df_dim*2, (3, 3), strides=(2,2), padding="same")(h0)
  h1 = tfa.layers.InstanceNormalization() (h1)
  h1 = tf.keras.layers.LeakyReLU() (h1)

  h2 = tf.keras.layers.Conv2D(df_dim*4, (3, 3), strides=(2,2), padding="same")(h1)
  h2 = tfa.layers.InstanceNormalization() (h2)
  h2 = tf.keras.layers.LeakyReLU() (h2)

  h3 = tf.keras.layers.Conv2D(df_dim*8, (3, 3), strides=(1,1), padding="same")(h2)
  h3 = tfa.layers.InstanceNormalization() (h3)
  h3 = tf.keras.layers.LeakyReLU() (h3)

  h4 = tf.keras.layers.Conv2D(segment_class, (3, 3), padding="same")(h3)
  h4 = tf.keras.layers.multiply([h4, mask])

  h4_mask = tf.math.reduce_sum(h4, axis=-1, keepdims=True)
  
  D_model = tf.keras.Model(inputs = [inputs, mask], outputs = h4_mask)

  return D_model

# Other functions 

def tf_kernel_prep_3d(kernel, n_channels):
    return np.tile(kernel, (n_channels, 1, 1)).swapaxes(0,1).swapaxes(1,2)

def tf_deriv(batch, ksize=3, padding='SAME'):
    n_ch = int(batch.get_shape().as_list()[3])
    gx = tf_kernel_prep_3d(np.array([[-1, 0, 1],
                                     [-2, 0, 2],
                                     [-1, 0, 1]]), n_ch)
    gy = tf_kernel_prep_3d(np.array([[-1,-2, -1],
                                     [ 0, 0, 0],
                                     [ 1, 2, 1]]), n_ch)
    kernel = tf.constant(np.stack([gx, gy], axis=-1), name="DerivKernel_image", dtype = np.float32)
    return tf.nn.depthwise_conv2d(batch, kernel, [1, 1, 1, 1], padding=padding, name="GradXY")

def abs_criterion(in_, target):
    return tf.math.reduce_mean(tf.abs(in_ - target))


def mae_criterion(in_, target):
    return tf.math.reduce_mean((in_-target)**2)


def sce_criterion(logits, labels):
    return tf.math.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=logits, labels=labels))

def gradloss_criterion(in_, target, weight):
    # print("gradloss_criterion")
    abs_deriv = tf.abs(tf.abs(tf_deriv(tf.convert_to_tensor(in_))) - tf.abs(tf_deriv(target)))
    abs_deriv = tf.math.reduce_mean(abs_deriv, axis=-1, keepdims=True)
    return tf.math.reduce_mean(tf.multiply(weight, abs_deriv))