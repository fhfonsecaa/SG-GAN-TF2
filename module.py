from __future__ import division

import numpy as np

import tensorflow as tf
import tensorflow_addons as tfa
# (unused) from tensorflow.keras import layers

# (unused) from ops import *
# (unused) from utils import *

OUTPUT_CHANNELS = 3

def downsample(filters, size, apply_batchnorm=True):
  initializer = tf.random_normal_initializer(0., 0.02)

  result = tf.keras.Sequential()
  result.add(
      tf.keras.layers.Conv2D(filters, size, strides=2, padding='same',
                             kernel_initializer=initializer, use_bias=False))

  if apply_batchnorm:
    result.add(tf.keras.layers.BatchNormalization())

  result.add(tf.keras.layers.LeakyReLU())

  return result

def upsample(filters, size, apply_dropout=False):
  initializer = tf.random_normal_initializer(0., 0.02)

  result = tf.keras.Sequential()
  result.add(
    tf.keras.layers.Conv2DTranspose(filters, size, strides=2,
                                    padding='same',
                                    kernel_initializer=initializer,
                                    use_bias=False))

  result.add(tf.keras.layers.BatchNormalization())

  if apply_dropout:
      result.add(tf.keras.layers.Dropout(0.5))

  result.add(tf.keras.layers.ReLU())

  return result

def generator_pix2pix():
  inputs = tf.keras.layers.Input(shape=(128,128,3,))

  down_stack = [
    downsample(64, 4, apply_batchnorm=False), # (bs, 128, 128, 64)
    downsample(128, 4), # (bs, 64, 64, 128)
    downsample(256, 4), # (bs, 32, 32, 256)
    downsample(512, 4), # (bs, 16, 16, 512)
    downsample(512, 4), # (bs, 8, 8, 512)
    downsample(512, 4), # (bs, 4, 4, 512)
    downsample(512, 4), # (bs, 2, 2, 512)
  ]

  up_stack = [
    upsample(512, 4, apply_dropout=True), # (bs, 2, 2, 1024)
    upsample(512, 4, apply_dropout=True), # (bs, 4, 4, 1024)
    upsample(512, 4, apply_dropout=True), # (bs, 8, 8, 1024)
    upsample(512, 4), # (bs, 16, 16, 1024)
    upsample(256, 4), # (bs, 32, 32, 512)
    upsample(128, 4), # (bs, 64, 64, 256)
    upsample(64, 4), # (bs, 128, 128, 128)
  ]

  initializer = tf.random_normal_initializer(0., 0.02)
  last = tf.keras.layers.Conv2DTranspose(OUTPUT_CHANNELS, 4,
                                         strides=2,
                                         padding='same',
                                         kernel_initializer=initializer,
                                         activation='tanh') # (bs, 256, 256, 3)

  x = inputs

  # Downsampling through the model
  skips = []
  for down in down_stack:
    x = down(x)
    skips.append(x)

  skips = reversed(skips[:-1])

  # Upsampling and establishing the skip connections
  for up, skip in zip(up_stack, skips):
    x = up(x)
    x = tf.keras.layers.Concatenate()([x, skip])

  x = last(x)

  return tf.keras.Model(inputs=inputs, outputs=x)

def discriminator_pix2pix():
  initializer = tf.random_normal_initializer(0., 0.02)

  inp = tf.keras.layers.Input(shape=(128,128,3,), name='input_image')
  tar = tf.keras.layers.Input(shape=(128,128,3,), name='target_image')

  x = tf.keras.layers.concatenate([inp, tar]) # (bs, 256, 256, channels*2)

  down1 = downsample(64, 4, False)(x) # (bs, 128, 128, 64)
  down2 = downsample(128, 4)(down1) # (bs, 64, 64, 128)
  down3 = downsample(256, 4)(down2) # (bs, 32, 32, 256)

  zero_pad1 = tf.keras.layers.ZeroPadding2D()(down3) # (bs, 34, 34, 256)
  conv = tf.keras.layers.Conv2D(512, 4, strides=1,
                                kernel_initializer=initializer,
                                use_bias=False)(zero_pad1) # (bs, 31, 31, 512)

  batchnorm1 = tf.keras.layers.BatchNormalization()(conv)

  leaky_relu = tf.keras.layers.LeakyReLU()(batchnorm1)

  zero_pad2 = tf.keras.layers.ZeroPadding2D()(leaky_relu) # (bs, 33, 33, 512)

  last = tf.keras.layers.Conv2D(1, 4, strides=1,
                                kernel_initializer=initializer)(zero_pad2) # (bs, 30, 30, 1)

  return tf.keras.Model(inputs=[inp, tar], outputs=last)

def generator_unet():
  print("generator_unet")
  gf_dim = 64
  output_c_dim = 3 
  is_training = True

  dropout_rate = 0.5 if is_training else 1.0
  
  # inputs = tf.keras.layers.Input(shape=(32,32,3,))
  inputs = tf.keras.layers.Input(shape=(128,128,3,))
  
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
  d8 = tf.keras.layers.Activation('tanh')(d8)
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
  inputs = tf.keras.layers.Input(shape=(128,128,3,),)
  
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
  image_height = 128 #32
  image_width = 128 #32

  inputs = tf.keras.layers.Input(shape=(image_height,image_width,3,))
  
  # mask = tf.keras.layers.Input(shape=(image_height,image_width,3,))
  mask = tf.keras.layers.Input(shape=(int(image_height/34), int(image_width/34), segment_class))

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
  
  h31 = tf.keras.layers.Conv2D(df_dim*8, (3, 3), strides=(2,2), padding="valid")(h3)
  h31 = tfa.layers.InstanceNormalization() (h31)
  h31 = tf.keras.layers.LeakyReLU() (h31)

  h32 = tf.keras.layers.Conv2D(df_dim*8, (3, 3), strides=(2,2), padding="valid")(h31)
  h32 = tfa.layers.InstanceNormalization() (h32)
  h32 = tf.keras.layers.LeakyReLU() (h32)
  
  h33 = tf.keras.layers.Conv2D(df_dim*8, (3, 3), strides=(1,1), padding="valid")(h32)
  h33 = tfa.layers.InstanceNormalization() (h33)
  h33 = tf.keras.layers.LeakyReLU() (h33)
  
  h4 = tf.keras.layers.Conv2D(segment_class, (3, 3), padding="same")(h33)
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