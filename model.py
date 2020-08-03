from __future__ import division
import os
import time
from glob import glob
import tensorflow as tf
import numpy as np
from collections import namedtuple

from module import generator_unet, generator_resnet, discriminator, mae_criterion, \
                    sce_criterion, tf_kernel_prep_3d, abs_criterion, gradloss_criterion

from utils import load_train_data, load_test_data, ImagePool, save_images

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
                                                                                args.input_nc + args.output_nc), name="real_A_and_B_images")
        self.seg_data = tf.keras.layers.Input(dtype=tf.dtypes.float32, shape=(args.image_height, args.image_width,
                                                                            args.input_nc + args.output_nc), name="seg_A_and_B_images")

        self.mask_A = tf.keras.layers.Input(dtype=tf.dtypes.float32, shape=(int(args.image_height/8), int(args.image_width/8), args.segment_class), name="mask_A")
        self.mask_B = tf.keras.layers.Input(dtype=tf.dtypes.float32, shape=(int(args.image_height/8), int(args.image_width/8), args.segment_class), name="mask_B")

        self.real_A =  self.real_data[:, :, :, :args.input_nc]                              # self.real_data[:, :, :, :self.input_c_dim]
        self.real_B = self.real_data[:, :, :, args.input_nc:args.input_nc + args.output_nc] # self.real_data[:, :, :, self.input_c_dim:self.input_c_dim + self.output_c_dim]
        self.seg_A = self.seg_data[:, :, :, :args.input_nc]                                 # self.seg_data[:, :, :, :self.input_c_dim]
        self.seg_B = self.seg_data[:, :, :, args.input_nc:args.input_nc + args.output_nc]   # self.seg_data[:, :, :, self.input_c_dim:self.input_c_dim + self.output_c_dim]

        #fake_A
        self.fake_A =  tf.keras.layers.Input(dtype=tf.dtypes.float32,            
                                             shape=(None, args.image_height, args.image_width, args.input_nc),
                                             name="fake_A_sample")             # tf.keras.layers.Input(dtype=tf.dtypes.float32, shape=(None, self.image_height, self.image_width, self.input_c_dim), name="fake_A_sample")
        #fake_B
        self.fake_B = tf.keras.layers.Input(dtype=tf.dtypes.float32,
                                            shape=(None, args.image_height, args.image_width, args.output_nc),
                                            name="fake_B_sample")              # tf.keras.layers.Input(dtype=tf.dtypes.float32, shape=(None, self.image_height, self.image_width, self.output_c_dim), name="fake_B_sample")

        # gradient kernel for seg
        # assume input_c_dim == output_c_dim
        self.kernels = []
        self.kernels.append( tf_kernel_prep_3d(np.array([[0,0,0],[-1,0,1],[0,0,0]]), args.input_nc) )     # self.kernels.append( tf_kernel_prep_3d(np.array([[0,0,0],[-1,0,1],[0,0,0]]), self.input_c_dim) )
        self.kernels.append( tf_kernel_prep_3d(np.array([[0,-1,0],[0,0,0],[0,1,0]]), args.input_nc) )     # self.kernels.append( tf_kernel_prep_3d(np.array([[0,-1,0],[0,0,0],[0,1,0]]), self.input_c_dim) )
        self.kernel = tf.constant(np.stack(self.kernels, axis=-1), name="DerivKernel_seg", dtype=np.float32)
        self.weighted_seg_A = []
        self.weighted_seg_B = []

    def generator_loss(self, DB_fake, DA_fake, args):
        # print("generator_loss")
        segA = tf.pad(self.seg_A, [[0, 0], [1, 1], [1, 1], [0, 0]], "REFLECT")
        segB = tf.pad(self.seg_B, [[0, 0], [1, 1], [1, 1], [0, 0]], "REFLECT")

        conved_seg_A = tf.abs(tf.nn.depthwise_conv2d(input=segA, filter=self.kernel, strides=[1, 1, 1, 1], padding="VALID", name="conved_seg_A"))
        conved_seg_B = tf.abs(tf.nn.depthwise_conv2d(segB, self.kernel, [1, 1, 1, 1], padding="VALID", name="conved_seg_B"))
        # change weighted_seg from (1.0, 0.0) to (0.9, 0.1) for soft gradient-sensitive loss
        self.weighted_seg_A = tf.abs(tf.sign(tf.math.reduce_sum(conved_seg_A, axis=-1, keepdims=True)))
        self.weighted_seg_B = tf.abs(tf.sign(tf.math.reduce_sum(conved_seg_B, axis=-1, keepdims=True)))
        
        g_loss_a2b = self.criterionGAN(DB_fake, tf.ones_like(DB_fake)) \
            + args.L1_lambda * abs_criterion(self.real_A, self.fake_A_) \
            + args.L1_lambda * abs_criterion(self.real_B, self.fake_B_) \
            + args.Lg_lambda * gradloss_criterion(self.real_A, self.fake_B, self.weighted_seg_A) \
            + args.Lg_lambda * gradloss_criterion(self.real_B, self.fake_A, self.weighted_seg_B)  
        
        g_loss_b2a = self.criterionGAN(DA_fake, tf.ones_like(DA_fake)) \
            + args.L1_lambda * abs_criterion(self.real_A, self.fake_A_) \
            + args.L1_lambda * abs_criterion(self.real_B, self.fake_B_) \
            + args.Lg_lambda * gradloss_criterion(self.real_A, self.fake_B, self.weighted_seg_A) \
            + args.Lg_lambda * gradloss_criterion(self.real_B, self.fake_A, self.weighted_seg_B)  
            
        g_loss = self.criterionGAN(DA_fake, tf.ones_like(DA_fake)) \
            + self.criterionGAN(DB_fake, tf.ones_like(DB_fake)) \
            + args.L1_lambda * abs_criterion(self.real_A, self.fake_A_) \
            + args.L1_lambda * abs_criterion(self.real_B, self.fake_B_) \
            + args.Lg_lambda * gradloss_criterion(self.real_A, self.fake_B, self.weighted_seg_A) \
            + args.Lg_lambda * gradloss_criterion(self.real_B, self.fake_A, self.weighted_seg_B)  
        
        return g_loss+g_loss_b2a+g_loss_a2b
        
    def discriminator_loss(self, DB_real, DA_real, DB_fake_sample, DA_fake_sample):
        # print("discriminator_loss")
        db_loss_real = self.criterionGAN(DB_real, tf.ones_like(DB_real))
        db_loss_fake = self.criterionGAN(DB_fake_sample, tf.zeros_like(DB_fake_sample))
        db_loss = (db_loss_real + db_loss_fake) / 2
        da_loss_real = self.criterionGAN(DA_real, tf.ones_like(DA_real))
        da_loss_fake = self.criterionGAN(DA_fake_sample, tf.zeros_like(DA_fake_sample))
        da_loss = (da_loss_real + da_loss_fake) / 2
        d_loss = da_loss + db_loss # self.d_loss = da_loss + db_loss
        
        return d_loss # self.d_loss
    
    # @tf.function
    def train_step (self, args):
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
        
            self.gen_loss = self.generator_loss(db_fake,da_fake, args)
            self.disc_loss = self.discriminator_loss(db_real, da_real, db_fake_sample, da_fake_sample)
            # print(self.gen_loss)

        generator_grads = gen_tape.gradient(self.gen_loss, self.generator.trainable_variables)
        discriminator_grads = disc_tape.gradient(self.disc_loss, self.discriminator.trainable_variables)
        
        self.g_optim.apply_gradients(zip(generator_grads, self.generator.trainable_variables))
        self.d_optim.apply_gradients(zip(discriminator_grads, self.discriminator.trainable_variables))

    def train(self, args):
        """Train SG-GAN"""
        
        lr  = 0.001 # self.lr = 0.001
        self.d_optim = tf.keras.optimizers.Adam(learning_rate=lr, beta_1=args.beta1)
        self.g_optim = tf.keras.optimizers.Adam(learning_rate=lr, beta_1=args.beta1)
        counter = 1
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
            # print("Episode Number: ", counter)
            dataA = glob('./datasets/{}/*.*'.format(args.dataset_dir + '/trainA'))  # glob('./datasets/{}/*.*'.format(self.dataset_dir + '/trainA'))
            dataB = glob('./datasets/{}/*.*'.format(args.dataset_dir + '/trainB'))  # glob('./datasets/{}/*.*'.format(self.dataset_dir + '/trainB'))
            np.random.shuffle(dataA)
            np.random.shuffle(dataB)
            batch_idxs = min(min(len(dataA), len(dataB)), args.train_size) // args.batch_size # self.batch_size
            # lr = args.lr if epoch < args.epoch_step else args.lr*(args.epoch-epoch)/(args.epoch-args.epoch_step)

            for idx in range(0, batch_idxs):
                # batch_files = list(zip(dataA[idx * self.batch_size:(idx + 1) * self.batch_size],
                #                       dataB[idx * self.batch_size:(idx + 1) * self.batch_size]))
                batch_files = list(zip(dataA[idx * args.batch_size:(idx + 1) * args.batch_size],
                                       dataB[idx * args.batch_size:(idx + 1) * args.batch_size]))
                
                batch_images = []
                batch_segs = []
                batch_seg_mask_A = []
                batch_seg_mask_B = []

                for batch_file in batch_files:
                    tmp_image, tmp_seg, tmp_seg_mask_A, tmp_seg_mask_B = load_train_data(batch_file, args.image_width, args.image_height,  num_seg_masks=args.segment_class) # num_seg_masks=self.segment_class)
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

                self.real_A = self.real_data[:, :, :, :args.input_nc]                               # self.real_A = self.real_data[:, :, :, :self.input_c_dim]
                self.real_B = self.real_data[:, :, :, args.input_nc:args.input_nc + args.output_nc] # self.real_B = self.real_data[:, :, :, self.input_c_dim:self.input_c_dim + self.output_c_dim]
                self.seg_A = self.seg_data[:, :, :, :args.input_nc]                                 # self.seg_A = self.seg_data[:, :, :, :self.input_c_dim]
                self.seg_B = self.seg_data[:, :, :, args.input_nc:args.input_nc + args.output_nc]   # self.seg_B = self.seg_data[:, :, :, self.input_c_dim:self.input_c_dim + self.output_c_dim]

                self.mask_A = batch_seg_mask_A
                self.mask_B = batch_seg_mask_B
                
                self.train_step(args)

                counter += 1
                print(("Epoch: [%2d] [%4d/%4d] time: %4.4f Gen_Loss: %f Disc_Loss: %f " % (
                    epoch, idx, batch_idxs, time.time() - start_time, self.gen_loss, self.disc_loss)))

                # if np.mod(counter, args.print_freq) == 1:
                #     self.test(args) # self.sample_model(args.sample_dir, epoch, idx, args)

                # if np.mod(counter, args.save_freq) == 2:
        self.save(args.checkpoint_dir, epoch)

    def save(self, checkpoint_dir, ep):
        """sggan_gene.model"""
        
        #### MODIFIED
        # model_dir = "%s" % self.dataset_dir
        # checkpoint_gene_dir = os.path.join(checkpoint_dir, model_dir)
        # model_name = "sggan_disc.model"
        # checkpoint_disc_dir = os.path.join(checkpoint_dir, model_dir)
        
        # if not os.path.exists(checkpoint_gene_dir):
        #     os.makedirs(checkpoint_gene_dir)

        # if not os.path.exists(checkpoint_disc_dir):
        #     os.makedirs(checkpoint_disc_dir)

        # self.generator.save(checkpoint_gene_dir)
        # self.discriminator.save(checkpoint_disc_dir)
        ####
        
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
        
        #### MODIFIED
        # model_dir = "%s" % self.dataset_dir
        # checkpoint_dir = os.path.join(checkpoint_dir, model_dir)
        ####
        
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
        
        #### MODIFIED
        # if g_ckpt and g_ckpt.model_checkpoint_path:
        #     g_ckpt_name = os.path.basename(g_ckpt.model_checkpoint_path)
        #     print("fail os.path.basename()")
        #     self.generator.load_weights(checkpoint_dir)
        #     print("fail load_weights()")
        #     self.discriminator.load_weights(checkpoint_dir)            
        #     self.saver.restore(self.sess, os.path.join(checkpoint_dir, ckpt_name))
        #     return True
        # else:
        #     return False
        ####

    def sample_model(self, sample_dir, epoch, idx, args):
        dataA = glob('./datasets/{}/*.*'.format(args.dataset_dir + '/testA'))     # glob('./datasets/{}/*.*'.format(self.dataset_dir + '/testA'))
        dataB = glob('./datasets/{}/*.*'.format(args.dataset_dir + '/testB'))     # glob('./datasets/{}/*.*'.format(self.dataset_dir + '/testB'))
        np.random.shuffle(dataA)
        np.random.shuffle(dataB)
        batch_files = list(zip(dataA[:args.batch_size], dataB[:args.batch_size])) # list(zip(dataA[:self.batch_size], dataB[:self.batch_size]))
        batch_images = []
        batch_segs = []
        
        for batch_file in batch_files:
            tmp_image, tmp_seg, _, _ = load_train_data(batch_file, args.img_width, args.img_height, num_seg_masks=args.segment_class, is_testing=True) # num_seg_masks=self.segment_class, is_testing=True)
            batch_images.append(tmp_image)
            batch_segs.append(tmp_seg)
            
        batch_images = np.array(batch_images).astype(np.float32)
        batch_segs = np.array(batch_segs).astype(np.float32)

        batch_img_A = batch_images[:, :, :, :args.input_nc]                                # batch_images[:, :, :, :self.input_c_dim]
        batch_img_B = batch_images[:, :, :, args.input_nc:args.input_nc + args.output_nc]  # batch_images[:, :, :, self.input_c_dim:self.input_c_dim + self.output_c_dim]
        
        fake_A, fake_B = self.generate_test_images(batch_img_A, batch_img_B)
        save_images(fake_A,  [args.batch_size, 1], # [self.batch_size, 1],
                    './{}/A_{:02d}_{:04d}_{}.jpg'.format(sample_dir, epoch, idx, batch_files[0][1].split("/")[-1].split(".")[0]))
        save_images(fake_B, [args.batch_size, 1],  # [self.batch_size, 1],
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
        """Test SG-GAN"""
        
        print(" [*] Running Test ...")
        
        if args.which_direction == 'AtoB':
            sample_files = glob('./datasets/{}/*.*'.format(args.dataset_dir + '/testA'))  # glob('./datasets/{}/*.*'.format(self.dataset_dir + '/testA'))
        elif args.which_direction == 'BtoA':
            sample_files = glob('./datasets/{}/*.*'.format(args.dataset_dir + '/testB'))  # glob('./datasets/{}/*.*'.format(self.dataset_dir + '/testB'))
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
            sample_image = [load_test_data(sample_file, args.image_width, args.image_height)]
            # [check print] # print("loaded test image:\n", sample_image)
            
            #### MODIFIED sample_image = np.array(sample_image).astype(np.float32) ####
            # Rescale pixels values into range [0,255]
            # (OK) rescaled_sample = [(255 * sample).astype(np.uint8) for sample in sample_image]
            rescaled_sample = [tf.image.convert_image_dtype(sample, np.uint8) for sample in sample_image]
            
            rescaled_sample = np.array(rescaled_sample).astype(np.uint8)
            sample_image = np.array(sample_image).astype(np.float32)
            # [check print] # print("converted test image:\n", sample_image)
            
            print("Type of sample_image: ", rescaled_sample.dtype)
            print("(?) Why gives warning if image is uint8")
            fake_A = self.generator(rescaled_sample)
            fake_B = self.generator(rescaled_sample)

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
