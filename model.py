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
        # Replaced placeholder with keras.layers.Input #
        self.real_data = tf.keras.layers.Input(dtype=tf.dtypes.float32, shape=(self.image_height, self.image_width,
                                                                                self.input_c_dim + self.output_c_dim), name="real_A_and_B_images")
        self.seg_data = tf.keras.layers.Input(dtype=tf.dtypes.float32, shape=(self.image_height, self.image_width,
                                                                            self.input_c_dim + self.output_c_dim), name="seg_A_and_B_images")

        self.mask_A = tf.keras.layers.Input(dtype=tf.dtypes.float32, shape=(int(self.image_height/8), int(self.image_width/8), self.segment_class), name="mask_A")
        self.mask_B = tf.keras.layers.Input(dtype=tf.dtypes.float32, shape=(int(self.image_height/8), int(self.image_width/8), self.segment_class), name="mask_B")

        self.real_A = self.real_data[:, :, :, :self.input_c_dim]
        self.real_B = self.real_data[:, :, :, self.input_c_dim:self.input_c_dim + self.output_c_dim]
        self.seg_A = self.seg_data[:, :, :, :self.input_c_dim]
        self.seg_B = self.seg_data[:, :, :, self.input_c_dim:self.input_c_dim + self.output_c_dim]

        #fake_A
        self.fake_A = tf.keras.layers.Input(dtype=tf.dtypes.float32, shape=(None, self.image_height, self.image_width, self.input_c_dim), name="fake_A_sample")
        #fake_B
        self.fake_B = tf.keras.layers.Input(dtype=tf.dtypes.float32, shape=(None, self.image_height, self.image_width, self.output_c_dim), name="fake_B_sample")

        # gradient kernel for seg
        # assume input_c_dim == output_c_dim
        self.kernels = []
        self.kernels.append( tf_kernel_prep_3d(np.array([[0,0,0],[-1,0,1],[0,0,0]]), self.input_c_dim) )
        self.kernels.append( tf_kernel_prep_3d(np.array([[0,-1,0],[0,0,0],[0,1,0]]), self.input_c_dim) )
        self.kernel = tf.constant(np.stack(self.kernels, axis=-1), name="DerivKernel_seg", dtype=np.float32)
        self.weighted_seg_A = []
        self.weighted_seg_B = []

    def generator_loss(self, DB_fake, DA_fake):
        # print("generator_loss")
        segA = tf.pad(self.seg_A, [[0, 0], [1, 1], [1, 1], [0, 0]], "REFLECT")
        segB = tf.pad(self.seg_B, [[0, 0], [1, 1], [1, 1], [0, 0]], "REFLECT")

        conved_seg_A = tf.abs(tf.nn.depthwise_conv2d(input=segA, filter=self.kernel, strides=[1, 1, 1, 1], padding="VALID", name="conved_seg_A"))
        conved_seg_B = tf.abs(tf.nn.depthwise_conv2d(segB, self.kernel, [1, 1, 1, 1], padding="VALID", name="conved_seg_B"))
        # change weighted_seg from (1.0, 0.0) to (0.9, 0.1) for soft gradient-sensitive loss
        self.weighted_seg_A = tf.abs(tf.sign(tf.math.reduce_sum(conved_seg_A, axis=-1, keepdims=True)))
        self.weighted_seg_B = tf.abs(tf.sign(tf.math.reduce_sum(conved_seg_B, axis=-1, keepdims=True)))
        
        g_loss_a2b = self.criterionGAN(DB_fake, tf.ones_like(DB_fake))\
            + self.L1_lambda * abs_criterion(self.real_A, self.fake_A_)\
            + self.L1_lambda * abs_criterion(self.real_B, self.fake_B_)\
            + self.Lg_lambda * gradloss_criterion(self.real_A, self.fake_B, self.weighted_seg_A)\
            + self.Lg_lambda * gradloss_criterion(self.real_B, self.fake_A, self.weighted_seg_B)

        g_loss_b2a = self.criterionGAN(DA_fake, tf.ones_like(DA_fake))\
            + self.L1_lambda * abs_criterion(self.real_A, self.fake_A_)\
            + self.L1_lambda * abs_criterion(self.real_B, self.fake_B_)\
            + self.Lg_lambda * gradloss_criterion(self.real_A, self.fake_B, self.weighted_seg_A)\
            + self.Lg_lambda * gradloss_criterion(self.real_B, self.fake_A, self.weighted_seg_B)

        g_loss = self.criterionGAN(DA_fake, tf.ones_like(DA_fake))\
            + self.criterionGAN(DB_fake, tf.ones_like(DB_fake))\
            + self.L1_lambda * abs_criterion(self.real_A, self.fake_A_)\
            + self.L1_lambda * abs_criterion(self.real_B, self.fake_B_)\
            + self.Lg_lambda * gradloss_criterion(self.real_A, self.fake_B, self.weighted_seg_A)\
            + self.Lg_lambda * gradloss_criterion(self.real_B, self.fake_A, self.weighted_seg_B)
        
        return g_loss+g_loss_b2a+g_loss_a2b
        
    def discriminator_loss(self, DB_real, DA_real, DB_fake_sample, DA_fake_sample):
        # print("discriminator_loss")
        db_loss_real = self.criterionGAN(DB_real, tf.ones_like(DB_real))
        db_loss_fake = self.criterionGAN(DB_fake_sample, tf.zeros_like(DB_fake_sample))
        db_loss = (db_loss_real + db_loss_fake) / 2
        da_loss_real = self.criterionGAN(DA_real, tf.ones_like(DA_real))
        da_loss_fake = self.criterionGAN(DA_fake_sample, tf.zeros_like(DA_fake_sample))
        da_loss = (da_loss_real + da_loss_fake) / 2
        self.d_loss = da_loss + db_loss
        
        return self.d_loss
    
    # @tf.function
    def train_step (self):
        # print("train_step")

        with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
            # print("GradientTape")

            self.fake_B = self.generator(self.real_A)
            self.fake_A_ = self.generator(self.fake_B)
            self.fake_A = self.generator(self.real_B)
            self.fake_B_ = self.generator(self.fake_A)
            
            db_fake = self.discriminator([self.fake_B, self.mask_A])
            da_fake = self.discriminator([self.fake_A, self.mask_B])
        
            db_real = self.discriminator([self.real_B, self.mask_B])
            da_real = self.discriminator([self.real_A, self.mask_A])
            db_fake_sample = self.discriminator([self.fake_B, self.mask_B])
            da_fake_sample = self.discriminator([self.fake_A, self.mask_A])
        
            self.gen_loss = self.generator_loss(db_fake,da_fake)
            self.disc_loss = self.discriminator_loss(db_real, da_real, db_fake_sample, da_fake_sample)
            # print(self.gen_loss)

        generator_grads = gen_tape.gradient(self.gen_loss, self.generator.trainable_variables)
        discriminator_grads = disc_tape.gradient(self.disc_loss, self.discriminator.trainable_variables)
        
        self.g_optim.apply_gradients(zip(generator_grads, self.generator.trainable_variables))
        self.d_optim.apply_gradients(zip(discriminator_grads, self.discriminator.trainable_variables))

    def train(self, args):
        """Train SG-GAN"""
        self.lr = 0.001
        self.d_optim = tf.keras.optimizers.Adam(learning_rate=self.lr, beta_1=args.beta1)
        self.g_optim = tf.keras.optimizers.Adam(learning_rate=self.lr, beta_1=args.beta1)
        counter = 1
        start_time = time.time()

        if args.continue_train and self.load(args.checkpoint_dir):
            print(" [*] Load SUCCESS")
        else:
            print(" [!] Load failed...")

        for epoch in range(args.epoch):
            # print("Episode Number: ", counter)
            dataA = glob('./datasets/{}/*.*'.format(self.dataset_dir + '/trainA'))
            dataB = glob('./datasets/{}/*.*'.format(self.dataset_dir + '/trainB'))
            np.random.shuffle(dataA)
            np.random.shuffle(dataB)
            batch_idxs = min(min(len(dataA), len(dataB)), args.train_size) // self.batch_size
            # lr = args.lr if epoch < args.epoch_step else args.lr*(args.epoch-epoch)/(args.epoch-args.epoch_step)

            for idx in range(0, batch_idxs):
                batch_files = list(zip(dataA[idx * self.batch_size:(idx + 1) * self.batch_size],
                                       dataB[idx * self.batch_size:(idx + 1) * self.batch_size]))
                batch_images = []
                batch_segs = []
                batch_seg_mask_A = []
                batch_seg_mask_B = []

                for batch_file in batch_files:
                    tmp_image, tmp_seg, tmp_seg_mask_A, tmp_seg_mask_B = load_train_data(batch_file, args.img_width, args.img_height, num_seg_masks=self.segment_class)
                    batch_images.append(tmp_image)
                    batch_segs.append(tmp_seg)

                    batch_seg_mask_A.append(tmp_seg_mask_A)
                    batch_seg_mask_B.append(tmp_seg_mask_B)
                    
                batch_images = np.array(batch_images).astype(np.float32)
                batch_segs = np.array(batch_segs).astype(np.float32)
                batch_seg_mask_A = np.array(batch_seg_mask_A).astype(np.float32)
                batch_seg_mask_B = np.array(batch_seg_mask_B).astype(np.float32)
                
                self.real_data = batch_images
                self.seg_data = batch_segs

                self.real_A = self.real_data[:, :, :, :self.input_c_dim]
                self.real_B = self.real_data[:, :, :, self.input_c_dim:self.input_c_dim + self.output_c_dim]
                self.seg_A = self.seg_data[:, :, :, :self.input_c_dim]
                self.seg_B = self.seg_data[:, :, :, self.input_c_dim:self.input_c_dim + self.output_c_dim]

                self.mask_A = batch_seg_mask_A
                self.mask_B = batch_seg_mask_B
                
                self.train_step()

                counter += 1
                print(("Epoch: [%2d] [%4d/%4d] time: %4.4f Gen_Loss: %f Disc_Loss: %f " % (
                    epoch, idx, batch_idxs, time.time() - start_time, self.gen_loss, self.disc_loss)))

                if np.mod(counter, args.print_freq) == 1:
                    self.test(args)
                    # self.sample_model(args.sample_dir, epoch, idx, args)

                if np.mod(counter, args.save_freq) == 2:
                    self.save(args.checkpoint_dir)

    # TODO: Save Checkpoint is not working proberly 
    def save(self, checkpoint_dir):
        """sggan_gene.model"""
        model_dir = "%s" % self.dataset_dir
        checkpoint_gene_dir = os.path.join(checkpoint_dir, model_dir)
        # model_name = "sggan_disc.model"
        checkpoint_disc_dir = os.path.join(checkpoint_dir, model_dir)

        if not os.path.exists(checkpoint_gene_dir):
            os.makedirs(checkpoint_gene_dir)

        if not os.path.exists(checkpoint_disc_dir):
            os.makedirs(checkpoint_disc_dir)

        self.generator.save(checkpoint_gene_dir)
        self.discriminator.save(checkpoint_disc_dir)

    # TODO: Load Checkpoint to be implemented 
    def load(self, checkpoint_dir):
        print(" [*] Reading checkpoint...")
        model_dir = "%s" % self.dataset_dir
        checkpoint_dir = os.path.join(checkpoint_dir, model_dir)
        ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
        print("I don't know: ", ckpt)

        if ckpt and ckpt.model_checkpoint_path:
            ckpt_name = os.path.basename(ckpt.model_checkpoint_path)
            self.generator.load_weights(checkpoint_dir)
            self.discriminator.load_weights(checkpoint_dir)            
            self.saver.restore(self.sess, os.path.join(checkpoint_dir, ckpt_name))
            return True
        else:
            return False

    def sample_model(self, sample_dir, epoch, idx, args):
        dataA = glob('./datasets/{}/*.*'.format(self.dataset_dir + '/testA'))
        dataB = glob('./datasets/{}/*.*'.format(self.dataset_dir + '/testB'))
        np.random.shuffle(dataA)
        np.random.shuffle(dataB)
        batch_files = list(zip(dataA[:self.batch_size], dataB[:self.batch_size]))
        batch_images = []
        batch_segs = []
        
        for batch_file in batch_files:
            tmp_image, tmp_seg, _, _ = load_train_data(batch_file, args.img_width, args.img_height, num_seg_masks=self.segment_class, is_testing=True)
            batch_images.append(tmp_image)
            batch_segs.append(tmp_seg)
            
        batch_images = np.array(batch_images).astype(np.float32)
        batch_segs = np.array(batch_segs).astype(np.float32)

        batch_img_A = batch_images[:, :, :, :self.input_c_dim]
        batch_img_B = batch_images[:, :, :, self.input_c_dim:self.input_c_dim + self.output_c_dim]
        
        fake_A, fake_B = self.generate_test_images(batch_img_A, batch_img_B)
        save_images(fake_A, [self.batch_size, 1],
                    './{}/A_{:02d}_{:04d}_{}.jpg'.format(sample_dir, epoch, idx, batch_files[0][1].split("/")[-1].split(".")[0]))
        save_images(fake_B, [self.batch_size, 1],
                    './{}/B_{:02d}_{:04d}_{}.jpg'.format(sample_dir, epoch, idx, batch_files[0][0].split("/")[-1].split(".")[0]))

    # @tf.function
    def generate_test_images(self, sample_imgA, sample_imgB):
        test_A = sample_imgA
        test_B = sample_imgB
        
        # direction == 'AtoB'
        testB = self.generator(test_A)
        # direction == 'BtoA'
        testA = self.generator(test_B)

        # testB = self.generator(test_A, self.options, True, name="generatorA2B")
        # testA = self.generator(test_B, self.options, True, name="generatorB2A")
        
        return testB, testA
        
    def test(self, args):
        print("Running Test")
        """Test SG-GAN"""
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
        
        for sample_file in sample_files:
            print('Processing image: ' + sample_file)
            sample_image = [load_test_data(sample_file, args.img_width, args.img_height)]
            sample_image = np.array(sample_image).astype(np.float32)            

            fake_A = self.generator(sample_image)
            fake_B = self.generator(sample_image)

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
