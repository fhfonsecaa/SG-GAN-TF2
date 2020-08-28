from __future__ import division
import os
import time
from glob import glob
import tensorflow as tf
import numpy as np
from collections import namedtuple

from module import generator_unet, generator_resnet, discriminator, mae_criterion, \
                    sce_criterion, tf_kernel_prep_3d, abs_criterion, gradloss_criterion

from utils import load_train_data, load_test_data, ImagePool, save_images, get_img, DataAugmentation, plot_tensors

import tensorflow as tf
import datetime, os

import metric
from metric import scores, dense_crf, scores_seg_fake, scores_seg_da_fake, scores_mask_sample_crf, scores_fake_mask_crf, scores_mask_fake_crf
import pandas as pd

generator_loss_metric = tf.keras.metrics.Mean(name='generator_loss_metric')
discriminator_loss_metric = tf.keras.metrics.Mean(name='discriminator_loss_metric')

logs_base_dir = "logs/"
if tf.io.gfile.exists(logs_base_dir):
  print('Path is there')
else:
  tf.io.gfile.makedirs(logs_base_dir)
  print('Path created')

logdir = os.path.join(logs_base_dir, datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
train_summary_writer = tf.summary.create_file_writer(logdir + '/train')
test_summary_writer = tf.summary.create_file_writer(logdir + '/test')

logs_base_dir = "logs/" # Because of the space in the My Drive


class sggan(object):
    def __init__(self, args):
        self.batch_size = args.batch_size
        self.image_width = args.image_width
        self.image_height = args.image_height
        self.input_c_dim = args.input_nc
        self.output_c_dim = args.output_nc
        self.L1_lambda = args.L1_lambda
        self.Lg_lambda = args.Lg_lambda
        self.dataset_dir = args.dataset_dir
        self.segment_class = args.segment_class
        self.alpha_recip = 1. / args.ratio_gan2seg if args.ratio_gan2seg > 0 else 0

        self.discriminator = discriminator()
        if args.use_resnet:
            self.generator = generator_resnet()
        else:
            self.generator = generator_unet()
        if args.use_lsgan:
            self.criterionGAN = mae_criterion
        else:
            self.criterionGAN = sce_criterion

        # tf.keras.utils.plot_model(self.discriminator, 'multi_input_and_output_model.png', show_shapes=True)
        # input("")

        OPTIONS = namedtuple('OPTIONS', 'batch_size image_height image_width \
                              gf_dim df_dim output_c_dim is_training segment_class')
        self.options = OPTIONS._make((args.batch_size, args.image_height, args.image_width,
                                      args.ngf, args.ndf, args.output_nc,
                                      args.phase == 'train', args.segment_class))

        self._build_model(args)
        self.pool = ImagePool(args.max_size)

        
        #### [ADDED] CHECKPOINT MANAGER
        self.lr = 0.001
        self.d_optim = tf.keras.optimizers.Adam(learning_rate=self.lr, beta_1=args.beta1)
        self.g_optim = tf.keras.optimizers.Adam(learning_rate=self.lr, beta_1=args.beta1)
        
        
        self.gen_ckpt = tf.train.Checkpoint(optimizer=self.g_optim, net=self.generator)
        self.disc_ckpt = tf.train.Checkpoint(optimizer=self.d_optim, net=self.discriminator)
        self.gen_ckpt_manager = tf.train.CheckpointManager(self.gen_ckpt, './checkpoint/gta/gen_ckpts', max_to_keep=3)
        self.disc_ckpt_manager = tf.train.CheckpointManager(self.disc_ckpt, './checkpoint/gta/disc_ckpts', max_to_keep=3)


    def _build_model(self, args):
        # Replaced placeholder with keras.layers.Input #
        self.real_data = tf.keras.layers.Input(dtype=tf.dtypes.float32, shape=(args.image_height, args.image_width,
                                                                                args.input_nc + args.output_nc), name="real_A_images")
        self.seg_data = tf.keras.layers.Input(dtype=tf.dtypes.float32, shape=(args.image_height, args.image_width,
                                                                            args.input_nc + args.output_nc), name="seg_A_images")

        self.mask_A = tf.keras.layers.Input(dtype=tf.dtypes.float32, shape=(int(args.image_height/8), int(args.image_width/8), args.segment_class), name="mask_A")

        self.real_A =  self.real_data[:, :, :, :args.input_nc]                              
        self.seg_A = self.seg_data[:, :, :, :args.input_nc]                                 

        #fake_A
        self.fake_A =  tf.keras.layers.Input(dtype=tf.dtypes.float32,            
                                             shape=(None, args.image_height, args.image_width, args.input_nc),
                                             name="fake_A_sample")

        # gradient kernel for seg
        # assume input_c_dim == output_c_dim
        self.kernels = []
        self.kernels.append( tf_kernel_prep_3d(np.array([[0,0,0],[-1,0,1],[0,0,0]]), args.input_nc) )
        self.kernels.append( tf_kernel_prep_3d(np.array([[0,-1,0],[0,0,0],[0,1,0]]), args.input_nc) )
        self.kernel = tf.constant(np.stack(self.kernels, axis=-1), name="DerivKernel_seg", dtype=np.float32)
        self.weighted_seg_A = []

    def generator_loss(self, DA_fake, args):
        # print("generator_loss")
        segA = tf.pad(self.seg_A, [[0, 0], [1, 1], [1, 1], [0, 0]], "REFLECT")

        conved_seg_A = tf.abs(tf.nn.depthwise_conv2d(input=segA, filter=self.kernel, strides=[1, 1, 1, 1], padding="VALID", name="conved_seg_A"))
        # change weighted_seg from (1.0, 0.0) to (0.9, 0.1) for soft gradient-sensitive loss
        self.weighted_seg_A = tf.abs(tf.sign(tf.math.reduce_sum(conved_seg_A, axis=-1, keepdims=True)))
            
        g_loss = self.criterionGAN(DA_fake, tf.ones_like(DA_fake)) \
            + args.L1_lambda * abs_criterion(self.real_A, self.fake_A) 
        
        return g_loss
        
    def discriminator_loss(self, DA_real, DA_fake_sample):
        # print("discriminator_loss")
        da_loss_real = self.criterionGAN(DA_real, tf.ones_like(DA_real))
        da_loss_fake = self.criterionGAN(DA_fake_sample, tf.zeros_like(DA_fake_sample))
        da_loss = (da_loss_real + da_loss_fake) / 2

        d_loss = da_loss 
        
        return d_loss

    def gen_loss_simple(self, DA_fake, args):
        # print("generator_loss")
        gan_loss  = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
            logits=DA_fake, labels=tf.ones_like(DA_fake)))
        seg_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=self.fake_A, labels=self.seg_A))
        # self.g_loss = self.alpha_recip * gan_loss + seg_loss
        
        return self.alpha_recip * gan_loss + seg_loss
    
    def disc_loss_simple(self, DA_real, DA_fake_sample):
        # print("discriminator_loss")
        d_loss_real = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
            logits=DA_real, labels=tf.ones_like(DA_real)))
        d_loss_fake = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
            logits=DA_fake_sample, labels=tf.zeros_like(DA_fake_sample)))
        # self.d_loss = self.d_loss_real + self.d_loss_fake        
        return d_loss_real + d_loss_fake
      
    def gen_loss_p2p(self, DA_fake, fake_A, seg_A):
        loss_object = tf.keras.losses.BinaryCrossentropy(from_logits=True)
        LAMBDA = 100
        # Losses computation
        gan_loss = loss_object(tf.ones_like(DA_fake), DA_fake)
        # mean absolute error
        l1_loss = tf.reduce_mean(tf.abs(seg_A - fake_A))
        total_gen_loss = gan_loss + (LAMBDA * l1_loss)
        
        return total_gen_loss
      
    def disc_loss_p2p(self, DA_real, DA_fake):
        loss_object = tf.keras.losses.BinaryCrossentropy(from_logits=True)
        real_loss = loss_object(tf.ones_like(DA_real), DA_real)
        generated_loss = loss_object(tf.zeros_like(DA_fake), DA_fake)
        total_disc_loss = real_loss + generated_loss
        
        return total_disc_loss

    # @tf.function
    def train_step (self, args):
        # print("train_step")

        with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
            # print("GradientTape")

            self.fake_A = self.generator(self.real_A)
            
            da_real = self.discriminator([self.seg_A, self.mask_A])
            da_fake = self.discriminator([self.fake_A, self.mask_A])
        
            da_fake_sample = self.discriminator([self.fake_A, self.mask_A])
        
            self.gen_loss = self.generator_loss(da_fake, args)
            self.disc_loss = self.discriminator_loss(da_real, da_fake_sample)
            # self.gen_loss = self.gen_loss_simple(da_fake, args)
            # self.disc_loss = self.disc_loss_simple(da_real, da_fake_sample)
            # self.gen_loss = self.gen_loss_p2p(da_fake, self.fake_A, self.seg_A)
            # self.disc_loss = self.disc_loss_p2p(da_real, da_fake_sample)

            # print(self.gen_loss)
                
        generator_loss_metric(self.gen_loss)
        discriminator_loss_metric(self.disc_loss)


        generator_grads = gen_tape.gradient(self.gen_loss, self.generator.trainable_variables)
        discriminator_grads = disc_tape.gradient(self.disc_loss, self.discriminator.trainable_variables)
        
        self.g_optim.apply_gradients(zip(generator_grads, self.generator.trainable_variables))
        self.d_optim.apply_gradients(zip(discriminator_grads, self.discriminator.trainable_variables))

    def train(self, args):
        """Train SG-GAN"""
        
        lr  = 0.001 # self.lr = 0.001
        self.d_optim = tf.keras.optimizers.Adam(learning_rate=lr, beta_1=args.beta1)
        self.g_optim = tf.keras.optimizers.Adam(learning_rate=lr, beta_1=args.beta1)
        start_time = time.time()

        if args.continue_train:
            print(" [*] Loading pretrained weights ...")
            if self.load(args.checkpoint_dir):
                print(" [*] Load SUCCESS")
            else:
                print(" [!] Load failed...")
        else:
            print(" [*] New training STARTED")

        for epoch in range(args.epoch):
            dataA = glob('./datasets/{}/*.*'.format(args.dataset_dir + '/trainA'))  # glob('./datasets/{}/*.*'.format(self.dataset_dir + '/trainA'))
            np.random.shuffle(dataA)
            batch_idxs = min(len(dataA), args.train_size) // args.batch_size # self.batch_size
            # lr = args.lr if epoch < args.epoch_step else args.lr*(args.epoch-epoch)/(args.epoch-args.epoch_step)

            augmenter = DataAugmentation()
            
            for idx in range(0, batch_idxs):
                batch_files = list(zip(dataA[idx * args.batch_size:(idx + 1) * args.batch_size]))
                
                batch_images = []
                batch_segs = []
                batch_seg_mask_A = []

                for batch_file in batch_files:
                    tmp_image, tmp_seg, tmp_seg_mask_A = load_train_data(batch_file, args.image_width, args.image_height,  num_seg_masks=args.segment_class, do_augment=False, augmenter=augmenter) # num_seg_masks=self.segment_class)
                    batch_images.append(tmp_image)
                    batch_segs.append(tmp_seg)
                    batch_seg_mask_A.append(tmp_seg_mask_A)

                    if (args.use_augmentation):
                        tmp_image, tmp_seg, tmp_seg_mask_A = load_train_data(batch_file, args.image_width, args.image_height,  num_seg_masks=args.segment_class, do_augment=True, augmenter=augmenter) # num_seg_masks=self.segment_class)
                        batch_images.append(tmp_image)
                        batch_segs.append(tmp_seg)
                        batch_seg_mask_A.append(tmp_seg_mask_A)
                
                batch_images = np.array(batch_images).astype(np.float32)
                batch_segs = np.array(batch_segs).astype(np.float32)
                batch_seg_mask_A = np.array(batch_seg_mask_A).astype(np.float32)
                
                self.real_data = batch_images
                self.seg_data = batch_segs

                self.real_A = self.real_data[:, :, :, :args.input_nc]
                self.seg_A = self.seg_data[:, :, :, :args.input_nc]

                self.mask_A = batch_seg_mask_A
                # print('test1',batch_images.size)
                # print('test1',batch_images.shape)
                
                self.train_step(args)

                print(("Epoch: [%2d] [%4d/%4d] time: %4.4f Gen_Loss: %f Disc_Loss: %f " % (
                    epoch, idx, batch_idxs, time.time() - start_time, self.gen_loss, self.disc_loss)))

            with train_summary_writer.as_default():
                fake = self.test_during_train(epoch, args)
                tf.summary.image('Segmentation Epoch {}'.format(epoch), fake, step=epoch)

                tf.summary.scalar('Generator Loss', generator_loss_metric.result(), step=epoch)
                tf.summary.scalar('Discriminator Loss', discriminator_loss_metric.result(), step=epoch)
                    
            generator_loss_metric.reset_states()
            discriminator_loss_metric.reset_states()
        self.save(args.checkpoint_dir, epoch)
        
    def get_labels (self, test_label, pred_img, crf=False):
        def swap_channels(tensor):
            return tf.transpose(tensor, [0,3,2,1])
        
        def crf_wrapper(true_image, pred_image):
            image = true_image
            image = (image * 255).astype(np.uint8)
            lt = image[0,:,:,:]
            print(lt.shape)
            lp = pred_image.numpy().transpose(0,3,2,1) #swap_channels(pred_image)
            lp = lp[0,:,:,:]
            print(lp.shape)
            # lp = (swap_channels(pred_image).numpy() * 255)
            # lp = lp[0,:,:,:]
            
            prob = dense_crf(lt, lp)
            prob = np.expand_dims(prob,axis=0)
            
            return image, prob #prob, image.transpose(0,3,2,1)
        
        if crf:
            lt, lp = crf_wrapper(test_label, pred_img)
            # lt = swap_channels(test_img).numpy()
        else:
            lt = test_label.numpy().transpose(0,3,2,1)
            lp = pred_img.numpy().transpose(0,3,2,1)
        
        return lt, lp
      
    def test_during_train(self, epoch, args):
        
        """Test SG-GAN"""        
        # print(" [*] Running Test ...")
        
        sample_files = glob('./datasets/{}/*.*'.format(args.dataset_dir + '/testA'))  # glob('./datasets/{}/*.*'.format(self.dataset_dir + '/testA'))
        
        preds1=[]; preds2=[]; preds3=[]; preds4=[]; preds5=[];
        gts1=[]; gts2=[]; gts3=[]; gts4=[]; gts5=[];
        
        fake_img = []
        actual_image = []
        output_images = []

        plot_labels = True
        
        for sample_file in sample_files:
            # print('Processing image: ' + sample_file)
            
            #### [MODIFIED] to test metric functions ####
            #### sample_image = [load_test_data(sample_file, args.image_width, args.image_height)]
            
            #### [CHANGES]
            sample_image, seg_image, seg_mask_64, seg_mask_8 = load_test_data(sample_file, args.image_width, args.image_height)
            sample_image = [sample_image]
            seg_image = [seg_image]
            # seg_maks_64 = [seg_mask_64]
            seg_mask_8 = [seg_mask_8]
            
            seg_image = np.array(seg_image).astype(np.float32)
            seg_mask_8 = np.array(seg_mask_8).astype(np.float32)
            seg_mask_64 = np.expand_dims(seg_mask_64, axis=0)
            ####
            
            
            rescaled_sample = [tf.image.convert_image_dtype(sample, np.uint8) for sample in sample_image]           
            rescaled_sample = np.array(rescaled_sample).astype(np.float32)
            sample_image = np.array(sample_image).astype(np.float32)
            
            # Get fake image
            fake_A = self.generator(rescaled_sample)
            fake_img = fake_A
            
            sample_image = (sample_image*2)-1

            image_path = os.path.join(args.test_dir, os.path.basename(sample_file))
            real_image_copy = os.path.join(args.test_dir, "real_" + os.path.basename(sample_file))
            # save_images(sample_image, [1, 1], real_image_copy)
            save_images(fake_img, [1, 1], image_path)
            
            # Get fake image
            actual_image = get_img(sample_image, [1, 1])
            fake_img = get_img(fake_A, [1, 1])
            # actual_image = np.array(actual_image).astype(np.uint8)
            
            output_images.append(fake_img)

            # Get da_fake discriminator output
            da_fake = self.discriminator([fake_A, seg_mask_8])
            # da_fake_rescaled = tf.image.convert_image_dtype(da_fake, np.uint8)
            
            # Get test, prediction labels
            # lp, lt = self.get_labels(actual_image, fake_img, crf=False)
            # preds += list(np.argmax(lp, axis=1))
            # gts += list(np.argmax(lt, axis=1))
            
            lt1, lp1 = scores_seg_fake(seg_image, fake_img)
            preds1 += list(lp1)
            gts1 += list(lt1)
            
            # lt2, lp2 = scores_mask_sample_crf(seg_mask_64, rescaled_sample)
            # preds2 += list(lp2)
            # gts2 += list(lt2)
            
            # lt3, lp3 = scores_fake_mask_crf(seg_mask_64, rescaled_sample, fake_img)
            # preds3 += list(lp3)
            # gts3 += list(lt3)
            
            # lt4, lp4 = scores_seg_da_fake(seg_image, da_fake)
            # preds4 += list(lp4)
            # gts4 += list(lt4)
            
            # lt5, lp5 = scores_mask_fake_crf(rescaled_sample, seg_mask_64, fake_img)
            # preds5 += list(lp5)
            # gts5 += list(lt5)
            
            # return fake_img, actual_image
            # yield fake_img, actual_image
        print("score")            
        score = scores(gts1, preds1, n_class=args.segment_class)
        score_df = pd.DataFrame(score)
        
        # score_crf = scores(gts2, preds2, n_class=args.segment_class)
        # score_crf_df = pd.DataFrame(score_crf)
        
        # score_crf_2 = scores(gts3, preds3, n_class=args.segment_class)
        # score_crf_2_df = pd.DataFrame(score_crf_2)
        
        # score_d = scores(gts4, preds4, n_class=args.segment_class)
        # score_d_df = pd.DataFrame(score_d)
        
        # score_crf_3 = scores(gts5, preds5, n_class=args.segment_class)
        # score_crf_3_df = pd.DataFrame(score_crf_3)
        
        print("\n[*] ------------")
        print("[*] Test scores:\n")
        
        with train_summary_writer.as_default():
            tf.summary.scalar('Overall Accuracy', score["Overall Acc"], step=epoch)
            tf.summary.scalar('Mean Accuracy', score["Mean Acc"], step=epoch)
            tf.summary.scalar('Frequency Weighted Accuracy', score["FreqW Acc"], step=epoch)
            tf.summary.scalar('Mean IoU', score["Mean IoU"], step=epoch)

        ########
        # if plot_labels:
        #     title="[*] Labels: seg_image | fake_img"
        #     name1="seg_image"
        #     name2="fake_image"
        #     for lt, lp in zip(gts1, preds1):
        #         plot_tensors(lt, lp, title, name1, name2)
            
        # print("---------------------------")
        # print("lt: seg_img || lp: fake_img")
        # print(score_df)
        
        # ########
        # if plot_labels:
        #     title="[*] Labels: seg_class_mask | crf(sample_image)"
        #     name1="seg_class_mask"
        #     name2="crf(sample_image, seg_class_mask)"
        #     for lt, lp in zip(gts2, preds2):
        #         plot_tensors(lt, lp, title, name1, name2)
            
        # print("---------------------------")
        # print("lt: seg_mask || lp: crf(test sample)")
        # print(score_crf_df)
        
        # ########
        # if plot_labels:
        #     title="[*] Labels: fake_img | crf(sample_image, seg_mask)"
        #     name1="fake_img"
        #     name2="crf(sample_image, seg_mask)"
        #     for lt, lp in zip(gts3, preds3):
        #         plot_tensors(lt, lp, title, name1, name2)
            
        # print("-------------------------------------")
        # print("lt: fake_img || lp: crf(sample_image, seg_mask)")
        # print(score_crf_2_df)
        
        # #########
        # if plot_labels:
        #     title="[*] Labels: seg_image | fake_img"
        #     name1="seg_image"
        #     name2="da_fake"
        #     for lt, lp in zip(gts4, preds4):
        #         plot_tensors(lt, lp, title, name1, name2)
            
        # print("----------------------------")
        # print("lt: seg_image || lp: da_fake")
        # print(score_d_df)
        
        # #########
        # if plot_labels:
        #     title="[*] Labels: seg_mask | crf(sample_image, fake_img)"
        #     name1="seg_mask"
        #     name2="crf(sample_image, fake_img)"
        #     for lt, lp in zip(gts5, preds5):
        #         plot_tensors(lt, lp, title, name1, name2)
            
        # print("----------------------------")
        # print("lt: seg_mask | lp: crf(sample_image, fake_img)")
        # print(score_crf_3_df)
        # print("Making multiple image tensor:", len(output_images))

        if(len(output_images) <= 1):
            return output_images[0]
        else:
            output_tensor = tf.concat([output_images[0], output_images[1]], axis=0)
            for i in range(2,len(output_images)):
                output_tensor = tf.concat([output_tensor, output_images[i]], axis=0)

            return output_tensor

    def save(self, checkpoint_dir, ep):
        """sggan_gene.model"""
      
        print(" [*] Saving checkpoint...")
        
        checkpoint_path = "%s/%s" % (checkpoint_dir,self.dataset_dir)
        gen_checkpoint_path = os.path.join(checkpoint_path, "gen/cp-{epoch:04d}.ckpt")  # "./checkpoint/gta/gen/cp-{epoch:04d}.ckpt"
        disc_checkpoint_path = os.path.join(checkpoint_path, "disc/cp-{epoch:04d}.ckpt") # "./checkpoint/gta/disc/cp-{epoch:04d}.ckpt"
        
        #### CHECK PRINT ####
        print("gen_checkpoint_path: %s" % gen_checkpoint_path)
        print("disc_checkpoint_path: %s" % disc_checkpoint_path)
        
        # Save generator model weights
        self.generator.save_weights(gen_checkpoint_path.format(epoch=ep))
        
        # Save discriminator model weights
        self.discriminator.save_weights(disc_checkpoint_path.format(epoch=ep))
        print(" [*] Checkpoints saved!")


    def load(self, checkpoint_dir):
        print(" [*] Reading checkpoint...")
        
        checkpoint_path = "%s/%s" % (checkpoint_dir,self.dataset_dir)
        gen_checkpoint_path = os.path.join(checkpoint_path, "gen/cp-{epoch:04d}.ckpt")  # "./checkpoint/gta/gen/cp-{epoch:04d}.ckpt"
        disc_checkpoint_path = os.path.join(checkpoint_path, "disc/cp-{epoch:04d}.ckpt") # "./checkpoint/gta/disc/cp-{epoch:04d}.ckpt"
        
        #### CHECK PRINT ####
        print("gen_checkpoint_path: %s" % gen_checkpoint_path)
        print("disc_checkpoint_path: %s" % disc_checkpoint_path)
        
        gen_checkpoint_dir = os.path.dirname(gen_checkpoint_path)
        disc_checkpoint_dir = os.path.dirname(disc_checkpoint_path)

        #### CHECK PRINT ####
        print("gen_checkpoint_dir: %s" % gen_checkpoint_dir)
        print("disc_checkpoint_dir: %s" % disc_checkpoint_dir)
        
        # Get latest training checkpoints 
        latest_g = tf.train.latest_checkpoint(gen_checkpoint_dir)
        latest_d = tf.train.latest_checkpoint(disc_checkpoint_dir)
        
        #### CHECK PRINT ####
        print("last_ckpt_gen: ", latest_g)
        print("last_ckpt_disc: ", latest_d)
        
        # Load generator and discriminator model weights
        if latest_g and latest_d:
            self.generator.load_weights(latest_g)
            self.discriminator.load_weights(latest_d)
            return True
        else:
            return False
        

    def sample_model(self, sample_dir, epoch, idx, args):
        dataA = glob('./datasets/{}/*.*'.format(args.dataset_dir + '/testA'))     # glob('./datasets/{}/*.*'.format(self.dataset_dir + '/testA'))
        np.random.shuffle(dataA)
        batch_files = list(zip(dataA[:args.batch_size]))
        batch_images = []
        batch_segs = []
        
        for batch_file in batch_files:
            tmp_image, tmp_seg, _  = load_train_data(batch_file, args.img_width, args.img_height, num_seg_masks=args.segment_class, is_testing=True) # num_seg_masks=self.segment_class, is_testing=True)
            batch_images.append(tmp_image)
            batch_segs.append(tmp_seg)
            
        batch_images = np.array(batch_images).astype(np.float32)
        batch_segs = np.array(batch_segs).astype(np.float32)

        batch_img_A = batch_images[:, :, :, :args.input_nc]                                # batch_images[:, :, :, :self.input_c_dim]
        
        fake_A = self.generate_test_images(batch_img_A)
        save_images(fake_A,  [args.batch_size, 1], # [self.batch_size, 1],
                    './{}/A_{:02d}_{:04d}_{}.jpg'.format(sample_dir, epoch, idx, batch_files[0][1].split("/")[-1].split(".")[0]))

    # @tf.function
    def generate_test_images(self, sample_imgA):
        test_A = sample_imgA
        testA = self.generator(test_A)

        return testA


    def test(self, args):
        """Test SG-GAN"""
        
        print(" [*] Running Test ...")
        
        sample_files = glob('./datasets/{}/*.*'.format(args.dataset_dir + '/testA'))  # glob('./datasets/{}/*.*'.format(self.dataset_dir + '/testA'))

        if self.load(args.checkpoint_dir):
            print(" [*] Load SUCCESS")
        else:
            print(" [!] Load failed...")
        
        for sample_file in sample_files:
            print('Processing image: ' + sample_file)
            sample_image = [load_test_data(sample_file, args.image_width, args.image_height)]
            # [check print] # print("loaded test image:\n", sample_image)
            
            #### MODIFIED sample_image = np.array(sample_image).astype(np.float32) ####
            # Rescale pixels values into range [0,255]
            # (OK) rescaled_sample = [(255 * sample).astype(np.uint8) for sample in sample_image]
            rescaled_sample = [tf.image.convert_image_dtype(sample, np.uint8) for sample in sample_image]
            
            rescaled_sample = np.array(rescaled_sample).astype(np.float32)
            sample_image = np.array(sample_image).astype(np.float32)
            # [check print] # print("converted test image:\n", sample_image)

            fake_A = self.generator(rescaled_sample)
            fake_img = fake_A
            
            image_path = os.path.join(args.test_dir, os.path.basename(sample_file))
            real_image_copy = os.path.join(args.test_dir, "real_" + os.path.basename(sample_file))
            save_images(sample_image, [1, 1], real_image_copy)
            save_images(fake_img, [1, 1], image_path)

