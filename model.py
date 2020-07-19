from __future__ import division
import os
import time
from glob import glob
import tensorflow as tf
import numpy as np
from collections import namedtuple

from module import *
from utils import *

class sggan(object):
    def __init__(self, args):
        self.batch_size = args.batch_size
        self.image_width = args.img_width
        self.image_height = args.img_height
        self.input_c_dim = args.input_nc
        self.output_c_dim = args.output_nc
        self.L1_lambda = args.L1_lambda
        self.Lg_lambda = args.Lg_lambda
        self.dataset_dir = args.dataset_dir
        self.segment_class = args.segment_class

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
        self.options = OPTIONS._make((args.batch_size, args.img_height, args.img_width,
                                      args.ngf, args.ndf, args.output_nc,
                                      args.phase == 'train', args.segment_class))

        self._build_model()
        self.pool = ImagePool(args.max_size)


    def _build_model(self):
        # MIGRATED TO TF2 #
        # self.real_data = tf.placeholder(tf.float32,
        #                                 [None, self.image_height, self.image_width,
        #                                  self.input_c_dim + self.output_c_dim],
        #                                 name='real_A_and_B_images')
        # self.seg_data = tf.placeholder(tf.float32,
        #                                 [None, self.image_height, self.image_width,
        #                                  self.input_c_dim + self.output_c_dim],
        #                                 name='seg_A_and_B_images')
        # self.mask_A = tf.placeholder(tf.float32, [None, self.image_height/8, self.image_width/8, self.segment_class], name='mask_A')
        # self.mask_B = tf.placeholder(tf.float32, [None, self.image_height/8, self.image_width/8, self.segment_class], name='mask_B')
        
        # self.real_A = self.real_data[:, :, :, :self.input_c_dim]
        # self.real_B = self.real_data[:, :, :, self.input_c_dim:self.input_c_dim + self.output_c_dim]
        # self.seg_A = self.seg_data[:, :, :, :self.input_c_dim]
        # self.seg_B = self.seg_data[:, :, :, self.input_c_dim:self.input_c_dim + self.output_c_dim]


        # gradient kernel for seg
        # assume input_c_dim == output_c_dim
        self.kernels = []
        self.kernels.append( tf_kernel_prep_3d(np.array([[0,0,0],[-1,0,1],[0,0,0]]), self.input_c_dim) )
        self.kernels.append( tf_kernel_prep_3d(np.array([[0,-1,0],[0,0,0],[0,1,0]]), self.input_c_dim) )
        self.kernel = tf.constant(np.stack(self.kernels, axis=-1), name="DerivKernel_seg", dtype=np.float32)
        self.weighted_seg_A = []
        self.weighted_seg_B = []
        
        # self.seg_A = tf.pad(self.seg_A, [[0, 0], [1, 1], [1, 1], [0, 0]], "REFLECT")
        # self.seg_B = tf.pad(self.seg_B, [[0, 0], [1, 1], [1, 1], [0, 0]], "REFLECT")
        # self.conved_seg_A = tf.abs(tf.nn.depthwise_conv2d(self.seg_A, self.kernel, [1, 1, 1, 1], padding="VALID", name="conved_seg_A"))
        # self.conved_seg_B = tf.abs(tf.nn.depthwise_conv2d(self.seg_B, self.kernel, [1, 1, 1, 1], padding="VALID", name="conved_seg_B"))
        
        # change weighted_seg from (1.0, 0.0) to (0.9, 0.1) for soft gradient-sensitive loss
        #### self.weighted_seg_A = tf.abs(tf.sign(tf.reduce_sum(self.conved_seg_A, axis=-1, keep_dims=True)))
        #### self.weighted_seg_B = tf.abs(tf.sign(tf.reduce_sum(self.conved_seg_B, axis=-1, keep_dims=True)))
        #self.weighted_seg_A = 0.9 * tf.abs(tf.sign(tf.reduce_sum(self.conved_seg_A, axis=-1, keep_dims=True))) + 0.1
        #self.weighted_seg_B = 0.9 * tf.abs(tf.sign(tf.reduce_sum(self.conved_seg_B, axis=-1, keep_dims=True))) + 0.1

        #self.weighted_seg_A = tf.sign(tf.reduce_sum(self.conved_seg_A, axis=-1, keep_dims=True))
        #self.weighted_seg_B = tf.sign(tf.reduce_sum(self.conved_seg_B, axis=-1, keep_dims=True))
        #self.weighted_seg_A = 0.9 * tf.sign(tf.reduce_sum(self.conved_seg_A, axis=-1, keep_dims=True)) + 0.1
        #self.weighted_seg_B = 0.9 * tf.sign(tf.reduce_sum(self.conved_seg_B, axis=-1, keep_dims=True)) + 0.1
        
        
        # Loss # # MIGRATED TO TF2 #
        # self.g_loss_a2b = self.criterionGAN(self.DB_fake, tf.ones_like(self.DB_fake)) \
        #     + self.L1_lambda * abs_criterion(self.real_A, self.fake_A_) \
        #     + self.L1_lambda * abs_criterion(self.real_B, self.fake_B_) \
        #     + self.Lg_lambda * gradloss_criterion(self.real_A, self.fake_B, self.weighted_seg_A) \
        #     + self.Lg_lambda * gradloss_criterion(self.real_B, self.fake_A, self.weighted_seg_B)
        # self.g_loss_b2a = self.criterionGAN(self.DA_fake, tf.ones_like(self.DA_fake)) \
        #     + self.L1_lambda * abs_criterion(self.real_A, self.fake_A_) \
        #     + self.L1_lambda * abs_criterion(self.real_B, self.fake_B_) \
        #     + self.Lg_lambda * gradloss_criterion(self.real_A, self.fake_B, self.weighted_seg_A) \
        #     + self.Lg_lambda * gradloss_criterion(self.real_B, self.fake_A, self.weighted_seg_B)
        # self.g_loss = self.criterionGAN(self.DA_fake, tf.ones_like(self.DA_fake)) \
        #     + self.criterionGAN(self.DB_fake, tf.ones_like(self.DB_fake)) \
        #     + self.L1_lambda * abs_criterion(self.real_A, self.fake_A_) \
        #     + self.L1_lambda * abs_criterion(self.real_B, self.fake_B_) \
        #     + self.Lg_lambda * gradloss_criterion(self.real_A, self.fake_B, self.weighted_seg_A) \
        #     + self.Lg_lambda * gradloss_criterion(self.real_B, self.fake_A, self.weighted_seg_B)

        
        # Loss # # MIGRATED TO TF2 #
        # self.db_loss_real = self.criterionGAN(self.DB_real, tf.ones_like(self.DB_real))
        # self.db_loss_fake = self.criterionGAN(self.DB_fake_sample, tf.zeros_like(self.DB_fake_sample))
        # self.db_loss = (self.db_loss_real + self.db_loss_fake) / 2
        # self.da_loss_real = self.criterionGAN(self.DA_real, tf.ones_like(self.DA_real))
        # self.da_loss_fake = self.criterionGAN(self.DA_fake_sample, tf.zeros_like(self.DA_fake_sample))
        # self.da_loss = (self.da_loss_real + self.da_loss_fake) / 2
        # self.d_loss = self.da_loss + self.db_loss

        # self.g_loss_a2b_sum = tf.summary.scalar("g_loss_a2b", self.g_loss_a2b)
        # self.g_loss_b2a_sum = tf.summary.scalar("g_loss_b2a", self.g_loss_b2a)
        # self.g_loss_sum = tf.summary.scalar("g_loss", self.g_loss)
        # self.g_sum = tf.summary.merge([self.g_loss_a2b_sum, self.g_loss_b2a_sum, self.g_loss_sum])
        # self.db_loss_sum = tf.summary.scalar("db_loss", self.db_loss)
        # self.da_loss_sum = tf.summary.scalar("da_loss", self.da_loss)
        # self.d_loss_sum = tf.summary.scalar("d_loss", self.d_loss)
        # self.db_loss_real_sum = tf.summary.scalar("db_loss_real", self.db_loss_real)
        # self.db_loss_fake_sum = tf.summary.scalar("db_loss_fake", self.db_loss_fake)
        # self.da_loss_real_sum = tf.summary.scalar("da_loss_real", self.da_loss_real)
        # self.da_loss_fake_sum = tf.summary.scalar("da_loss_fake", self.da_loss_fake)
        # self.d_sum = tf.summary.merge(
        #     [self.da_loss_sum, self.da_loss_real_sum, self.da_loss_fake_sum,
        #      self.db_loss_sum, self.db_loss_real_sum, self.db_loss_fake_sum,
        #      self.d_loss_sum]
        # )

        # MIGRATED TO TF2 #
        # self.test_A = tf.placeholder(tf.float32,
        #                              [None, self.image_height, self.image_width,
        #                               self.input_c_dim], name='test_A')
        # self.test_B = tf.placeholder(tf.float32,
        #                              [None, self.image_height, self.image_width,
        #                               self.output_c_dim], name='test_B')
        # self.testB = self.generator(self.test_A, self.options, True, name="generatorA2B")
        # self.testA = self.generator(self.test_B, self.options, True, name="generatorB2A")


    def generator_loss(self, DB_fake, DA_fake, real_A, real_B, fake_A, fake_B, seg_A, seg_B):
        segA = tf.pad(self.seg_A, [[0, 0], [1, 1], [1, 1], [0, 0]], "REFLECT")
        segB = tf.pad(self.seg_B, [[0, 0], [1, 1], [1, 1], [0, 0]], "REFLECT")
        conved_seg_A = tf.abs(tf.nn.depthwise_conv2d(self.segA, self.kernel, [1, 1, 1, 1], padding="VALID", name="conved_seg_A"))
        conved_seg_B = tf.abs(tf.nn.depthwise_conv2d(self.segB, self.kernel, [1, 1, 1, 1], padding="VALID", name="conved_seg_B"))
        # change weighted_seg from (1.0, 0.0) to (0.9, 0.1) for soft gradient-sensitive loss
        self.weighted_seg_A = tf.abs(tf.sign(tf.reduce_sum(conved_seg_A, axis=-1, keep_dims=True)))
        self.weighted_seg_B = tf.abs(tf.sign(tf.reduce_sum(conved_seg_B, axis=-1, keep_dims=True)))
        
        g_loss_a2b = self.criterionGAN(DB_fake, tf.ones_like(DB_fake)) \
            + self.L1_lambda * abs_criterion(real_A, fake_A_) \
            + self.L1_lambda * abs_criterion(real_B, fake_B_) \
            + self.Lg_lambda * gradloss_criterion(real_A, fake_B, self.weighted_seg_A) \
            + self.Lg_lambda * gradloss_criterion(real_B, fake_A, self.weighted_seg_B)
        g_loss_b2a = self.criterionGAN(DA_fake, tf.ones_like(DA_fake)) \
            + self.L1_lambda * abs_criterion(real_A, fake_A_) \
            + self.L1_lambda * abs_criterion(real_B, fake_B_) \
            + self.Lg_lambda * gradloss_criterion(real_A, fake_B, self.weighted_seg_A) \
            + self.Lg_lambda * gradloss_criterion(real_B, fake_A, self.weighted_seg_B)
        self.g_loss = self.criterionGAN(DA_fake, tf.ones_like(DA_fake)) \
            + self.criterionGAN(DB_fake, tf.ones_like(DB_fake)) \
            + self.L1_lambda * abs_criterion(real_A, fake_A_) \
            + self.L1_lambda * abs_criterion(real_B, fake_B_) \
            + self.Lg_lambda * gradloss_criterion(real_A, fake_B, self.weighted_seg_A) \
            + self.Lg_lambda * gradloss_criterion(real_B, fake_A, self.weighted_seg_B)
        
        g_loss_a2b_sum = tf.summary.scalar("g_loss_a2b", g_loss_a2b)
        g_loss_b2a_sum = tf.summary.scalar("g_loss_b2a", g_loss_b2a)
        g_loss_sum = tf.summary.scalar("g_loss", g_loss)
        
        self.g_sum = tf.summary.merge([g_loss_a2b_sum, g_loss_b2a_sum, g_loss_sum])
        
        return g_loss
        
    def discriminator_loss(self, DB_real, DA_real, DB_fake_sample, DA_fake_sample):
        db_loss_real = self.criterionGAN(DB_real, tf.ones_like(DB_real))
        db_loss_fake = self.criterionGAN(DB_fake_sample, tf.zeros_like(DB_fake_sample))
        db_loss = (db_loss_real + db_loss_fake) / 2
        da_loss_real = self.criterionGAN(DA_real, tf.ones_like(DA_real))
        da_loss_fake = self.criterionGAN(DA_fake_sample, tf.zeros_like(DA_fake_sample))
        da_loss = (da_loss_real + da_loss_fake) / 2
        self.d_loss = da_loss + db_loss
        
        db_loss_sum = tf.summary.scalar("db_loss", db_loss)
        da_loss_sum = tf.summary.scalar("da_loss", da_loss)
        d_loss_sum = tf.summary.scalar("d_loss", d_loss)
        db_loss_real_sum = tf.summary.scalar("db_loss_real", db_loss_real)
        db_loss_fake_sum = tf.summary.scalar("db_loss_fake", db_loss_fake)
        da_loss_real_sum = tf.summary.scalar("da_loss_real", da_loss_real)
        da_loss_fake_sum = tf.summary.scalar("da_loss_fake", da_loss_fake)
        
        self.d_sum = tf.summary.merge(
            [da_loss_sum, da_loss_real_sum, da_loss_fake_sum,
             db_loss_sum, db_loss_real_sum, db_loss_fake_sum,
             d_loss_sum]
        )
        
        return d_loss
    
    @tf.function
    def train_step (self, real_A , real_B, mask_A, mask_B, seg_A, seg_B):
        
        with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
            fake_B = self.generator(real_A)
            fake_A_ = self.generator(fake_B)
            fake_A = self.generator(real_B)
            fake_B_ = self.generator(fake_A)
            
            db_fake = self.discriminator([fake_B, mask_A])
            da_fake = self.discriminator([fake_A, mask_B])
        
            db_real = self.discriminator([real_B, mask_B])
            da_real = self.discriminator([real_A, mask_A])
            db_fake = self.discriminator([fake_B, mask_B])
            da_fake_sample = self.discriminator([fake_A, mask_A])
        
            gen_loss = self.generator_loss(db_fake,da_fake, real_A, real_B, fake_A, fake_B, seg_A, seg_B)
            disc_loss = self.discriminator_loss(db_real, da_real, db_fake_sample, da_fake_sample)
        
        generator_grads = gen_tape.gradient(gen_loss, self.generator.trainable_variables)
        discriminator_grads = disc_tape.gradient(disc_loss, self.discriminator.trainable_variables)
        
        self.g_optim.apply_gradients(zip(generator_grads, self.g_vars))
        self.d_optim.apply_gradients(zip(discriminator_grads, self.g_vars))

    def train(self, args):
        """Train SG-GAN"""
        self.lr = 0.001
        self.d_optim = tf.keras.optimizers.Adam(learning_rate=self.lr, beta_1=args.beta1)
            # .minimize(self.d_loss, var_list=self.discriminator.trainable_variables)
        self.g_optim = tf.keras.optimizers.Adam(learning_rate=self.lr, beta_1=args.beta1)
            # .minimize(self.g_loss, var_list=self.generator.trainable_variables)

        # init_op = tf.global_variables_initializer()
        # self.sess.run(init_op)
        # self.writer = tf.summary.FileWriter("./logs", self.sess.graph)

        counter = 1
        start_time = time.time()

        if args.continue_train and self.load(args.checkpoint_dir):
            print(" [*] Load SUCCESS")
        else:
            print(" [!] Load failed...")
        for epoch in range(args.epoch):
            dataA = glob('./datasets/{}/*.*'.format(self.dataset_dir + '/trainA'))
            dataB = glob('./datasets/{}/*.*'.format(self.dataset_dir + '/trainB'))
            np.random.shuffle(dataA)
            np.random.shuffle(dataB)
            batch_idxs = min(min(len(dataA), len(dataB)), args.train_size) // self.batch_size
            lr = args.lr if epoch < args.epoch_step else args.lr*(args.epoch-epoch)/(args.epoch-args.epoch_step)

            for idx in range(0, batch_idxs):
                batch_files = list(zip(dataA[idx * self.batch_size:(idx + 1) * self.batch_size],
                                       dataB[idx * self.batch_size:(idx + 1) * self.batch_size]))
                #batch_images = []
                #batch_segs = []
                batch_img_A = []
                batch_img_B = []
                batch_seg_A = []
                batch_seg_B = []
                batch_seg_mask_A = []
                batch_seg_mask_B = []
                for batch_file in batch_files:
                    #tmp_image, tmp_seg, tmp_seg_mask_A, tmp_seg_mask_B = load_train_data(batch_file, args.img_width, args.img_height, num_seg_masks=self.segment_class)
                    tmp_imgA, tmp_imgB, tmp_segA, tmp_segB, tmp_seg_mask_A, tmp_seg_mask_B = load_train_data(batch_file, args.img_width, args.img_height, num_seg_masks=self.segment_class)
                    #batch_images.append(tmp_image)
                    #batch_segs.append(tmp_seg)
                    batch_img_A.append(tmp_imgA)
                    batch_img_B.append(tmp_imgB)
                    batch_seg_A.append(tmp_segA)
                    batch_seg_B.append(tmp_segB)
                    batch_seg_mask_A.append(tmp_seg_mask_A)
                    batch_seg_mask_B.append(tmp_seg_mask_B)
                    
                #batch_images = np.array(batch_images).astype(np.float32)
                #batch_segs = np.array(batch_segs).astype(np.float32)
                batch_img_A = np.array(batch_img_A).astype(np.float32)
                batch_img_B = np.array(batch_img_B).astype(np.float32)
                batch_seg_A = np.array(batch_seg_A).astype(np.float32)
                batch_seg_B = np.array(batch_seg_B).astype(np.float32)
                batch_seg_mask_A = np.array(batch_seg_mask_A).astype(np.float32)
                batch_seg_mask_B = np.array(batch_seg_mask_B).astype(np.float32)
                
                # MIGRATED TO TF2 #
                # Update G network and record fake outputs
                # fake_A, fake_B, fake_A_mask, fake_B_mask, _, summary_str = self.sess.run(
                #     [self.fake_A, self.fake_B, self.mask_B, self.mask_A, self.g_optim, self.g_sum],
                #     feed_dict={self.real_data: batch_images, self.lr: lr, self.seg_data: batch_segs,
                #     self.mask_A: batch_seg_mask_A, self.mask_B: batch_seg_mask_B})
                # self.writer.add_summary(summary_str, counter)
                # [fake_A, fake_B, fake_A_mask, fake_B_mask] = self.pool([fake_A, fake_B, fake_A_mask, fake_B_mask])
                # Update D network
                # _, summary_str = self.sess.run(
                #     [self.d_optim, self.d_sum],
                #     feed_dict={self.real_data: batch_images,
                #                self.fake_A_sample: fake_A,
                #                self.fake_B_sample: fake_B,
                #                self.mask_A_sample: fake_A_mask,
                #                self.mask_B_sample: fake_B_mask,
                #                self.mask_A: batch_seg_mask_A,
                #                self.mask_B: batch_seg_mask_B,
                #                self.lr: lr})
                # self.writer.add_summary(summary_str, counter)
                
                self.train_step(batch_img_A, batch_img_B, batch_seg_mask_A, batch_seg_mask_B, batch_seg_A, batch_seg_B)
                counter += 1
                print(("Epoch: [%2d] [%4d/%4d] time: %4.4f" % (
                    epoch, idx, batch_idxs, time.time() - start_time)))

                if np.mod(counter, args.print_freq) == 1:
                    self.sample_model(args.sample_dir, epoch, idx)

                if np.mod(counter, args.save_freq) == 2:
                    self.save(args.checkpoint_dir)

    def save(self, checkpoint_dir):
        model_name = "sggan_gene.model"
        model_dir = "%s" % self.dataset_dir
        checkpoint_gene_dir = os.path.join(checkpoint_dir, model_dir)
        model_name = "sggan_disc.model"
        checkpoint_disc_dir = os.path.join(checkpoint_dir, model_dir)

        if not os.path.exists(checkpoint_gene_dir):
            os.makedirs(checkpoint_gene_dir)

        if not os.path.exists(checkpoint_disc_dir):
            os.makedirs(checkpoint_disc_dir)

        self.generator.save(checkpoint_gene_dir)
        self.discriminator.save(checkpoint_disc_dir)

    # TODO
    # def load(self, checkpoint_dir):
    #     print(" [*] Reading checkpoint...")

    #     model_dir = "%s" % self.dataset_dir
    #     checkpoint_dir = os.path.join(checkpoint_dir, model_dir)

    #     ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
    #     if ckpt and ckpt.model_checkpoint_path:
    #         ckpt_name = os.path.basename(ckpt.model_checkpoint_path)
    #         self.saver.restore(self.sess, os.path.join(checkpoint_dir, ckpt_name))
    #         return True
    #     else:
    #         return False

    def sample_model(self, sample_dir, epoch, idx):
        dataA = glob('./datasets/{}/*.*'.format(self.dataset_dir + '/testA'))
        dataB = glob('./datasets/{}/*.*'.format(self.dataset_dir + '/testB'))
        np.random.shuffle(dataA)
        np.random.shuffle(dataB)
        batch_files = list(zip(dataA[:self.batch_size], dataB[:self.batch_size]))
        batch_images = []
        batch_segs = []
        for batch_file in batch_files:
            # MIGRATED TO TF2 #
            # tmp_image, tmp_seg, tmp_seg_mask_A, tmp_seg_mask_B = load_train_data(batch_file, self.image_width, self.image_height, is_testing=True, num_seg_masks=self.segment_class)
            # batch_images.append(tmp_image)
            # batch_segs.append(tmp_seg)
            
            tmp_imgA, tmp_imgB, tmp_segA, temp_segB, tmp_seg_mask_A, tmp_seg_mask_B = load_train_data(batch_file, self.image_width, self.image_height, is_testing=True, num_seg_masks=self.segment_class)
            batch_img_A.append(tmp_imgA)
            batch_img_B.append(tmp_imgB)
            batch_seg_A.append(tmp_segA)
            batch_seg_B.append(tmp_segB)
            batch_seg_mask_A.append(tmp_seg_mask_A)
            batch_seg_mask_B.append(tmp_seg_mask_B)
        
        # MIGRATED TO TF2 #
        # batch_images = np.array(batch_images).astype(np.float32)
        # batch_segs = np.array(batch_segs).astype(np.float32)
            
        batch_img_A = np.array(batch_img_A).astype(np.float32)
        batch_img_B = np.array(batch_img_B).astype(np.float32)
        batch_seg_A = np.array(batch_seg_A).astype(np.float32)
        batch_seg_B = np.array(batch_seg_B).astype(np.float32)
        batch_seg_mask_A = np.array(batch_seg_mask_A).astype(np.float32)
        batch_seg_mask_B = np.array(batch_seg_mask_B).astype(np.float32)
        
        # MIGRATED TO TF2 #
        # fake_A, fake_B = self.sess.run(
        #     [self.fake_A, self.fake_B],
        #     feed_dict={self.real_data: batch_images, self.seg_data: batch_segs}
        # )
        
        fake_A, fake_B = self.generate_test_images(batch_img_A, batch_img_B, batch_seg_A, batch_seg_B)
        save_images(fake_A, [self.batch_size, 1],
                    './{}/A_{:02d}_{:04d}_{}.jpg'.format(sample_dir, epoch, idx, batch_files[0][1].split("/")[-1].split(".")[0]))
        save_images(fake_B, [self.batch_size, 1],
                    './{}/B_{:02d}_{:04d}_{}.jpg'.format(sample_dir, epoch, idx, batch_files[0][0].split("/")[-1].split(".")[0]))

    @tf.function
    def generate_test_images(self, sample_imgA, sample_imgB, seg_A, seg_B):
        segA = tf.pad(seg_A, [[0, 0], [1, 1], [1, 1], [0, 0]], "REFLECT")
        segB = tf.pad(seg_B, [[0, 0], [1, 1], [1, 1], [0, 0]], "REFLECT")
        conved_seg_A = tf.abs(tf.nn.depthwise_conv2d(segA, self.kernel, [1, 1, 1, 1], padding="VALID", name="conved_seg_A"))
        conved_seg_B = tf.abs(tf.nn.depthwise_conv2d(segB, self.kernel, [1, 1, 1, 1], padding="VALID", name="conved_seg_B"))
        # change weighted_seg from (1.0, 0.0) to (0.9, 0.1) for soft gradient-sensitive loss
        self.weighted_seg_A = tf.abs(tf.sign(tf.reduce_sum(conved_seg_A, axis=-1, keep_dims=True)))
        self.weighted_seg_B = tf.abs(tf.sign(tf.reduce_sum(conved_seg_B, axis=-1, keep_dims=True)))
        
        test_A = sample_imgA
        test_B = sample_imgB
        
        # direction == 'AtoB'
        testB = self.generator(test_A, self.options, True, name="generatorA2B")
        # direction == 'BtoA'
        testA = self.generator(test_B, self.options, True, name="generatorB2A")
        
        return testB, testA
        
    def test(self, args):
        """Test SG-GAN"""
        
        # init_op = tf.global_variables_initializer()
        # self.sess.run(init_op)
        
        if args.which_direction == 'AtoB':
            sample_files = glob('./datasets/{}/*.*'.format(self.dataset_dir + '/testA'))
        elif args.which_direction == 'BtoA':
            sample_files = glob('./datasets/{}/*.*'.format(self.dataset_dir + '/testB'))
        else:
            raise Exception('--which_direction must be AtoB or BtoA')

        if self.load(args.checkpoint_dir):
            print(" [*] Load SUCCESS")
        else:
            print(" [!] Load failed...")

        # write html for visual comparison
        index_path = os.path.join(args.test_dir, '{0}_index.html'.format(args.which_direction))
        index = open(index_path, "w")
        index.write("<html><body><table><tr>")
        index.write("<th>name</th><th>input</th><th>output</th></tr>")
        
        # MIGRATED TO TF2 #
        # out_var, in_var = (self.testB, self.test_A) if args.which_direction == 'AtoB' else (
        #     self.testA, self.test_B)
        
        for sample_file in sample_files:
            print('Processing image: ' + sample_file)
            # sample_image = [load_test_data(sample_file, args.img_width, args.img_height)]
            # sample_image = np.array(sample_image).astype(np.float32)
            
            sample_imgA, sample_imgB, sample_segA, sample_segB, _ , _ = load_train_data(sample_file, args.img_width, args.img_height, is_testing=True)
            sample_imgA = np.array(batch_img_A).astype(np.float32)
            sample_imgB = np.array(batch_img_B).astype(np.float32)
            sample_segA = np.array(batch_seg_A).astype(np.float32)
            sample_segB = np.array(batch_seg_B).astype(np.float32)
            
            # MIGRATED TO TF2 #
            # fake_img = self.sess.run(out_var, feed_dict={in_var: sample_image})
            fake_A, fake_B = self.generate_test_images(sample_imgA, sample_imgB)
            if args.which_direction == 'AtoB':
                fake_img = fake_A
            else:
                fake_img = fake_B
            
            image_path = os.path.join(args.test_dir, os.path.basename(sample_file))
            real_image_copy = os.path.join(args.test_dir, "real_" + os.path.basename(sample_file))
            save_images(sample_image, [1, 1], real_image_copy)
            save_images(fake_img, [1, 1], image_path)
            index.write("<td>%s</td>" % os.path.basename(image_path))
            index.write("<td><img src='%s'></td>" % (real_image_copy if os.path.isabs(real_image_copy) else (
                '..' + os.path.sep + real_image_copy)))
            index.write("<td><img src='%s'></td>" % (image_path if os.path.isabs(image_path) else (
                '..' + os.path.sep + image_path)))
            index.write("</tr>")
        index.close()
