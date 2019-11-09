import tensorflow as tf

import os.path as osp
import numpy as np
import time

from tflearn import is_training
from latent_3d_points.gan import GAN
from latent_3d_points.neural_net import MODEL_SAVER_ID


class wgan_translator(GAN):

    def __init__(self, name, init_lr, lam, cycleLossWeight, featureLossWeight,  npoints, sizeBNeck, \
                 discriminator, generator, ae_AB, ae_epoch, \
                 batch_size=128, beta=0.5, gen_kwargs={}, disc_kwargs={}, graph=None):

        GAN.__init__(self, name, graph)

        self.name = name
        self.init_lr = init_lr
        self.lam = lam
        self.cycleLossWeight = cycleLossWeight
        self.featureLossWeight = featureLossWeight

        self.npoints = npoints
        self.sizeBNeck = sizeBNeck
        self.discriminator = discriminator
        self.generator = generator

        self.ae_AB = ae_AB
        self.ae_epoch = ae_epoch
        self.batch_size = batch_size
        self.beta = beta

        self.MODEL_SAVER_ID =  MODEL_SAVER_ID


        with tf.variable_scope(self.name):

            self.input_A_code = tf.placeholder(tf.float32, shape=[batch_size, sizeBNeck[0] ])
            self.input_B_code = tf.placeholder(tf.float32, shape=[batch_size, sizeBNeck[0] ])
            self.real_A_code = tf.placeholder(tf.float32, shape=[batch_size, sizeBNeck[0] ])
            self.real_B_code = tf.placeholder(tf.float32, shape=[batch_size, sizeBNeck[0] ])

            self.input_A_pc = tf.placeholder(tf.float32, shape=[batch_size, npoints[0], 3 ])
            self.input_B_pc = tf.placeholder(tf.float32, shape=[batch_size, npoints[0], 3 ])
            self.real_A_pc = tf.placeholder(tf.float32, shape=[batch_size, npoints[0], 3 ])
            self.real_B_pc = tf.placeholder(tf.float32, shape=[batch_size, npoints[0], 3 ])

            ## A-to-B and B-to-B
            with tf.variable_scope('generator_AB2B') as scope:
                self.generator_out_A2B_code = self.generator(self.input_A_code, self.sizeBNeck, scope=scope, **gen_kwargs)
            with tf.variable_scope('generator_AB2B', reuse=True)  as scope:
                self.generator_out_B2B_code = self.generator(self.input_B_code, self.sizeBNeck, scope=scope, **gen_kwargs)

            ## B-to-A and A-to-A
            with tf.variable_scope('generator_AB2A') as scope:
                self.generator_out_B2A_code = self.generator(self.input_B_code, self.sizeBNeck, scope=scope, **gen_kwargs)
            with tf.variable_scope('generator_AB2A', reuse=True)  as scope:
                self.generator_out_A2A_code = self.generator(self.input_A_code, self.sizeBNeck, scope=scope, **gen_kwargs)

            ## cycle output
            with tf.variable_scope('generator_AB2B', reuse=True)  as scope:
                self.generator_out_B2A2B_code = self.generator(self.generator_out_B2A_code, self.sizeBNeck, scope=scope, **gen_kwargs)
            with tf.variable_scope('generator_AB2A', reuse=True)  as scope:
                self.generator_out_A2B2A_code = self.generator(self.generator_out_A2B_code, self.sizeBNeck, scope=scope, **gen_kwargs)

        self.generator_out_A2B_pc   =   ae_AB.decode_layers(ae_AB.name, self.generator_out_A2B_code,   reuse=False)
        self.generator_out_B2B_pc   =   ae_AB.decode_layers(ae_AB.name, self.generator_out_B2B_code,   reuse=True)
        self.generator_out_B2A_pc   =   ae_AB.decode_layers(ae_AB.name, self.generator_out_B2A_code,   reuse=True)
        self.generator_out_A2A_pc   =   ae_AB.decode_layers(ae_AB.name, self.generator_out_A2A_code,   reuse=True)
        self.generator_out_B2A2B_pc   =   ae_AB.decode_layers(ae_AB.name, self.generator_out_B2A2B_code,   reuse=True)
        self.generator_out_A2B2A_pc   =   ae_AB.decode_layers(ae_AB.name, self.generator_out_A2B2A_code,   reuse=True)

        with tf.variable_scope(self.name):

            ################################# GAN losses:  AB -> B
            with tf.variable_scope('discriminator_B_code') as scope:
                self.real_prob_B_code, self.real_logit_B_code = self.discriminator(self.real_B_code, scope=scope, **disc_kwargs)
            with tf.variable_scope('discriminator_B_code', reuse=True)  as scope:
                self.synthetic_prob_A2B_code, self.synthetic_logit_A2B_code = self.discriminator(self.generator_out_A2B_code, reuse=True, scope=scope,  **disc_kwargs)

            self.loss_d_B_code     =   tf.reduce_mean(self.synthetic_logit_A2B_code) - tf.reduce_mean(self.real_logit_B_code) 
            self.loss_g_A2B_code   =   -tf.reduce_mean( self.synthetic_logit_A2B_code )			

            # Compute gradient penalty at interpolated points
            ndims = self.real_B_code.get_shape().ndims
            alpha = tf.random_uniform(shape=[self.batch_size] + [1] * (ndims - 1), minval=0., maxval=1.)
            differences = self.generator_out_A2B_code - self.real_B_code
            interpolates = self.real_B_code + (alpha * differences)
            with tf.variable_scope('discriminator_B_code', reuse=True)  as scope:
                gradients = tf.gradients(self.discriminator(interpolates, reuse=True, scope=scope, **disc_kwargs)[1], [interpolates])[0]
            # Reduce over all but the first dimension
            slopes = tf.sqrt(tf.reduce_sum(tf.square(gradients), axis=list(range(1, ndims) )))
            self.gradient_penalty_B = tf.reduce_mean((slopes - 1.) ** 2) * self.lam

            ################################# GAN losses:  AB -> A
            with tf.variable_scope('discriminator_A_code') as scope:
                self.real_prob_A_code, self.real_logit_A_code = self.discriminator(self.real_A_code, scope=scope, **disc_kwargs)
            with tf.variable_scope('discriminator_A_code', reuse=True)  as scope:
                self.synthetic_prob_B2A_code, self.synthetic_logit_B2A_code = self.discriminator(self.generator_out_B2A_code, reuse=True, scope=scope,  **disc_kwargs)

            self.loss_d_A_code     =   tf.reduce_mean(self.synthetic_logit_B2A_code) - tf.reduce_mean(self.real_logit_A_code) 
            self.loss_g_B2A_code   =   -tf.reduce_mean( self.synthetic_logit_B2A_code )

            # Compute gradient penalty at interpolated points
            ndims = self.real_A_code.get_shape().ndims
            alpha = tf.random_uniform(shape=[self.batch_size] + [1] * (ndims - 1), minval=0., maxval=1.)
            differences = self.generator_out_B2A_code - self.real_A_code
            interpolates = self.real_A_code + (alpha * differences)
            with tf.variable_scope('discriminator_A_code', reuse=True)  as scope:
                gradients = tf.gradients(self.discriminator(interpolates, reuse=True, scope=scope, **disc_kwargs)[1], [interpolates])[0]
            # Reduce over all but the first dimension
            slopes = tf.sqrt(tf.reduce_sum(tf.square(gradients), axis=list(range(1, ndims)  )))
            self.gradient_penalty_A = tf.reduce_mean((slopes - 1.) ** 2) * self.lam

            ############### cycle LOSS  ###############
            self.loss_c_A2B2A_code = tf.reduce_mean( tf.abs(self.input_A_code - self.generator_out_A2B2A_code)) * self.cycleLossWeight
            self.loss_c_B2A2B_code = tf.reduce_mean( tf.abs(self.input_B_code - self.generator_out_B2A2B_code)) * self.cycleLossWeight
            self.loss_Cycle_code = self.loss_c_A2B2A_code + self.loss_c_B2A2B_code

            ############### feature Loss  ###############
            diff_A2A = tf.abs(self.input_A_code - self.generator_out_A2A_code)
            diff_B2B = tf.abs(self.input_B_code - self.generator_out_B2B_code)
            self.loss_feature_A2A_code = tf.reduce_mean( diff_A2A ) * self.featureLossWeight
            self.loss_feature_B2B_code = tf.reduce_mean( diff_B2B ) * self.featureLossWeight
            self.loss_feature_code = self.loss_feature_A2A_code + self.loss_feature_B2B_code


            self.generatorLosses = self.loss_g_A2B_code  + self.loss_g_B2A_code  + self.loss_feature_code  + self.loss_Cycle_code


            train_vars = tf.trainable_variables()
            d_params_A_code   = [v for v in train_vars if v.name.startswith(name + '/discriminator_A_code/')]
            d_params_B_code   = [v for v in train_vars if v.name.startswith(name + '/discriminator_B_code/')]
            g_params_AB2A = [v for v in train_vars if v.name.startswith(name + '/generator_AB2A/')]
            g_params_AB2B = [v for v in train_vars if v.name.startswith(name + '/generator_AB2B/')]

            ###############  Config optimizer
            self.learning_rate = tf.train.exponential_decay(
                self.init_lr,  # base learning rate
                self.epoch,  # global_var indicating the number of steps
                100,  # step size
                0.5,  # decay rate
                staircase=True
            )
            self.learning_rate = tf.maximum(self.learning_rate, 5e-4)

            self.opt_d_B_code    = self.optimizer(self.learning_rate, self.beta, self.loss_d_B_code + self.gradient_penalty_B,   d_params_B_code)
            self.opt_d_A_code    = self.optimizer(self.learning_rate, self.beta, self.loss_d_A_code + self.gradient_penalty_A,   d_params_A_code)

            self.opt_Generators  = self.optimizer(self.learning_rate, self.beta, self.generatorLosses, g_params_AB2A + g_params_AB2B )

            self.saver = tf.train.Saver(tf.global_variables(), max_to_keep=None)
            self.init = tf.global_variables_initializer()

            # Launch the session
            config = tf.ConfigProto()
            config.gpu_options.allow_growth = True
            self.sess = tf.Session(config=config)
            self.sess.run(self.init)


        ############### load pre-trained Autoencoders
        ae_AB_model_path = osp.join(ae_AB.configuration.train_dir, MODEL_SAVER_ID + '-' + str(int(self.ae_epoch)))
        ae_AB_varlist = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=ae_AB.name )
        saver_aeA = tf.train.Saver( var_list=ae_AB_varlist )
        saver_aeA.restore(self.sess, ae_AB_model_path )
        #print(  ae_AB_varlist  )



    def translate_code( self, test_data, direction, batch_size ):

        n_examples = test_data.num_examples
        n_batches = int(n_examples / batch_size)

        generate_input = None
        generate_output = None

        for iter in range(n_batches):
            feed_data_pc, _, feed_data_code = test_data.next_batch(batch_size)

            if direction == 'A2B':
                feed_dict = {  self.input_A_pc: feed_data_pc,  self.input_A_code: feed_data_code }
                output = self.sess.run([self.generator_out_A2B_code], feed_dict=feed_dict)

            elif direction == 'B2B':
                feed_dict = {  self.input_B_pc: feed_data_pc,  self.input_B_code: feed_data_code }
                output = self.sess.run([self.generator_out_B2B_code], feed_dict=feed_dict)

            elif direction == 'B2A':
                feed_dict = {self.input_B_pc: feed_data_pc, self.input_B_code: feed_data_code}
                output = self.sess.run([self.generator_out_B2A_code], feed_dict=feed_dict)

            elif direction == 'A2A':
                feed_dict = {self.input_A_pc: feed_data_pc, self.input_A_code: feed_data_code}
                output = self.sess.run([self.generator_out_A2A_code], feed_dict=feed_dict)

            output = np.squeeze(output)

            if iter == 0:
                generate_input = feed_data_code
                generate_output = output
            else:
                generate_input  = np.concatenate((generate_input, feed_data_code), axis=0)
                generate_output = np.concatenate((generate_output, output), axis=0)

        return generate_input, generate_output



    def translate_PointClouds( self, test_data, direction, batch_size, onlyOnebatch=False ):

        n_examples = test_data.num_examples
        n_batches = int(n_examples / batch_size)

        if(onlyOnebatch):
            n_batches = 1

        generate_input = None
        generate_output = None

        for iter in range(n_batches):
            feed_data_pc, _, feed_data_code = test_data.next_batch(batch_size)

            if direction == 'A2B':
                feed_dict = {  self.input_A_pc: feed_data_pc,  self.input_A_code: feed_data_code }
                output = self.sess.run([self.generator_out_A2B_pc], feed_dict=feed_dict)

            elif direction == 'B2B':
                feed_dict = {  self.input_B_pc: feed_data_pc,  self.input_B_code: feed_data_code }
                output = self.sess.run([self.generator_out_B2B_pc], feed_dict=feed_dict)

            elif direction == 'B2A':
                feed_dict = {  self.input_B_pc: feed_data_pc,  self.input_B_code: feed_data_code }
                output = self.sess.run([self.generator_out_B2A_pc], feed_dict=feed_dict)

            elif direction == 'A2A':
                feed_dict = {  self.input_A_pc: feed_data_pc,  self.input_A_code: feed_data_code }
                output = self.sess.run([self.generator_out_A2A_pc], feed_dict=feed_dict)

            output = np.squeeze(output)

            if iter == 0:
                generate_input = feed_data_pc
                generate_output = output
            else:
                generate_input  = np.concatenate((generate_input, feed_data_pc), axis=0)
                generate_output = np.concatenate((generate_output, output), axis=0)

        return generate_input, generate_output



    def _single_epoch_train(self, train_data_A, train_data_B, discriminator_boost=2, generator_boost=1 ):


        train_data_A.shuffle_data()
        train_data_B.shuffle_data()

        n_examples = min( train_data_A.num_examples,  train_data_B.num_examples )
        epoch_loss_d_A_code = 0.
        epoch_loss_d_B_code = 0.
        epoch_loss_g_A2B_code = 0.
        epoch_loss_g_B2A_code = 0.
        epoch_loss_feature_B2B_code = 0.
        epoch_loss_feature_A2A_code = 0.
        epoch_loss_c_A2B2A_code = 0.0
        epoch_loss_c_B2A2B_code = 0.0
        epoch_gradient_penalty_A = 0.
        epoch_gradient_penalty_B = 0.
        current_learning_rate = 0.0

        batch_size = self.batch_size
        n_batches = int(n_examples / batch_size)
        #print("n_batches = ", n_batches)

        iterations_for_epoch = int( n_batches / max( discriminator_boost, generator_boost ) )
        #print("iterations_for_epoch = ", iterations_for_epoch)

        start_time = time.time()
        is_training(True, session=self.sess)
        try:
            # Loop over all batches
            for _ in range(iterations_for_epoch):

                feed_dict = None

                for _ in range(discriminator_boost):

                    feed_real_A_pc, _, feed_real_A_code = train_data_A.next_batch(batch_size)
                    feed_real_B_pc, _, feed_real_B_code = train_data_B.next_batch(batch_size)

                    feed_input_A_pc, _, feed_input_A_code = train_data_A.next_batch(batch_size)
                    feed_input_B_pc, _, feed_input_B_code = train_data_B.next_batch(batch_size)

                    feed_dict = {self.real_A_pc:  feed_real_A_pc,  self.real_A_code: feed_real_A_code, \
                                 self.real_B_pc:  feed_real_B_pc,  self.real_B_code: feed_real_B_code,   \
                                 self.input_A_pc: feed_input_A_pc,  self.input_A_code: feed_input_A_code,   \
                                 self.input_B_pc: feed_input_B_pc,  self.input_B_code: feed_input_B_code  }


                    _ = self.sess.run([self.opt_d_A_code ], feed_dict=feed_dict)
                    _ = self.sess.run([self.opt_d_B_code ], feed_dict=feed_dict)



                for _ in range(generator_boost):

                    feed_real_A_pc, _, feed_real_A_code = train_data_A.next_batch(batch_size)
                    feed_real_B_pc, _, feed_real_B_code = train_data_B.next_batch(batch_size)

                    feed_input_A_pc    =  feed_real_A_pc
                    feed_input_A_code  =  feed_real_A_code
                    feed_input_B_pc    =  feed_real_B_pc
                    feed_input_B_code  =  feed_real_B_code

                    feed_dict = {self.real_A_pc:  feed_real_A_pc,  self.real_A_code: feed_real_A_code, \
                                 self.real_B_pc:  feed_real_B_pc,  self.real_B_code: feed_real_B_code,   \
                                 self.input_A_pc: feed_input_A_pc,  self.input_A_code: feed_input_A_code,   \
                                 self.input_B_pc: feed_input_B_pc,  self.input_B_code: feed_input_B_code  }

                    _ = self.sess.run( [self.opt_Generators ], feed_dict=feed_dict)



                loss_d_B_code, loss_g_A2B_code,   loss_feature_B2B_code, loss_c_A2B2A_code, gradient_penalty_B  = self.sess.run([ \
                                               self.loss_d_B_code, self.loss_g_A2B_code,  \
                                               self.loss_feature_B2B_code, self.loss_c_A2B2A_code, self.gradient_penalty_B ],  feed_dict=feed_dict)

                loss_d_A_code, loss_g_B2A_code,   loss_feature_A2A_code, loss_c_B2A2B_code, gradient_penalty_A  = self.sess.run([
                                               self.loss_d_A_code,  self.loss_g_B2A_code, \
                                               self.loss_feature_A2A_code, self.loss_c_B2A2B_code, self.gradient_penalty_A ],  feed_dict=feed_dict)

                current_learning_rate = self.sess.run([self.learning_rate], feed_dict=feed_dict)

                epoch_loss_d_B_code += loss_d_B_code
                epoch_loss_g_A2B_code += loss_g_A2B_code
                epoch_loss_feature_B2B_code += loss_feature_B2B_code

                epoch_loss_d_A_code += loss_d_A_code
                epoch_loss_g_B2A_code += loss_g_B2A_code
                epoch_loss_feature_A2A_code += loss_feature_A2A_code

                epoch_loss_c_A2B2A_code += loss_c_A2B2A_code
                epoch_loss_c_B2A2B_code += loss_c_B2A2B_code

                epoch_gradient_penalty_A += gradient_penalty_A
                epoch_gradient_penalty_B += gradient_penalty_B


            is_training(False, session=self.sess)
        except Exception:
            raise
        finally:
            is_training(False, session=self.sess)

        epoch_loss_d_B_code /= iterations_for_epoch
        epoch_loss_g_A2B_code /= iterations_for_epoch
        epoch_loss_feature_B2B_code /= iterations_for_epoch

        epoch_loss_d_A_code /= iterations_for_epoch
        epoch_loss_g_B2A_code /= iterations_for_epoch
        epoch_loss_feature_A2A_code /= iterations_for_epoch

        epoch_loss_c_A2B2A_code /= iterations_for_epoch
        epoch_loss_c_B2A2B_code /= iterations_for_epoch

        epoch_gradient_penalty_A /= iterations_for_epoch
        epoch_gradient_penalty_B /= iterations_for_epoch

        duration = time.time() - start_time


        lossTuple = (epoch_loss_d_A_code, \
                      epoch_loss_d_B_code, \
                      epoch_loss_g_A2B_code, \
                      epoch_loss_g_B2A_code, \
                      epoch_loss_feature_B2B_code, \
                      epoch_loss_feature_A2A_code, \
                      epoch_loss_c_A2B2A_code, \
                      epoch_loss_c_B2A2B_code, \
                      epoch_gradient_penalty_A, \
                      epoch_gradient_penalty_B )

        otherTuple = (duration, current_learning_rate)

        return lossTuple, otherTuple

