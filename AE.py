import warnings
import os.path as osp
import tensorflow as tf
import numpy as np
import time

from tflearn import is_training

from in_out import create_dir
from general_utils import iterate_in_chunks
from latent_3d_points.neural_net import Neural_Net, MODEL_SAVER_ID


try:    
    from latent_3d_points.structural_losses.tf_nndistance import nn_distance
    from latent_3d_points.structural_losses.tf_approxmatch import approx_match, match_cost
except:
    print('External Losses (Chamfer-EMD) cannot be loaded. Please install them first.')
    exit()
    

class AutoEncoder(Neural_Net):
    '''
    An Auto-Encoder for point-clouds.
    '''

    def __init__(self, name, configuration, graph=None):
        
        c = configuration
        self.configuration = c
        self.name = name

        Neural_Net.__init__(self, name, graph)
        

        self.n_input = c.n_input
        self.n_output = c.n_output

        self.batch_size=c.batch_size

        in_shape  = [c.batch_size] + self.n_input
        out_shape = [c.batch_size] + self.n_output

        with tf.variable_scope(name):
            self.x = tf.placeholder(tf.float32, in_shape)
            self.gt = self.x

            self.z = c.encoder(self.x, **c.encoder_args)
			
            assert self.z.get_shape()[1]==256

            zerovector = tf.constant(0.0, dtype=tf.float32, shape=[self.z.get_shape()[0], 64])

            with tf.variable_scope('sharedDecoder') as scope:
                print(scope)
                subcode1 = tf.concat( [self.z[:,0:64], zerovector, zerovector, zerovector] , axis=1 )
                layer1 = c.decoder(subcode1, nameprefix='branch_5decoder', scope=scope, reuse=False, **c.decoder_args)

            with tf.variable_scope('sharedDecoder', reuse=True) as scope:
                print(scope)
                subcode2 = tf.concat( [zerovector, self.z[:,64:128], zerovector, zerovector] , axis=1 )
                layer2 = c.decoder(subcode2, nameprefix='branch_5decoder', scope=scope,  reuse=True,  **c.decoder_args)

            with tf.variable_scope('sharedDecoder', reuse=True) as scope:
                print(scope)
                subcode3 = tf.concat( [zerovector, zerovector, self.z[:,128:192], zerovector] , axis=1 )
                layer3 = c.decoder( subcode3, nameprefix='branch_5decoder', scope=scope, reuse=True,  **c.decoder_args)

            with tf.variable_scope('sharedDecoder', reuse=True) as scope:
                print(scope)
                subcode4 = tf.concat( [zerovector, zerovector, zerovector, self.z[:,192:256]] , axis=1 )
                layer4 = c.decoder(subcode4, nameprefix='branch_5decoder', scope=scope, reuse=True,  **c.decoder_args)

            with tf.variable_scope('sharedDecoder', reuse=True) as scope:
                print(scope)
                layer5 = c.decoder(self.z, nameprefix='branch_5decoder', scope=scope, reuse=True,  **c.decoder_args)


            self.x_b1 = tf.reshape(layer1, [-1, self.n_output[0], self.n_output[1]])
            self.x_b2 = tf.reshape(layer2, [-1, self.n_output[0], self.n_output[1]])
            self.x_b3 = tf.reshape(layer3, [-1, self.n_output[0], self.n_output[1]])
            self.x_b4 = tf.reshape(layer4, [-1, self.n_output[0], self.n_output[1]])
            self.x_b5 = tf.reshape(layer5, [-1, self.n_output[0], self.n_output[1]])
            

            self.x_all = tf.concat([self.x_b1,self.x_b2,self.x_b3,self.x_b4,self.x_b5], 2)

            self.x_reconstr = self.x_b5

            self.saver = tf.train.Saver(tf.global_variables(), max_to_keep=c.saver_max_to_keep)

            self._create_loss()
            self._setup_optimizer()


            # GPU configuration
            if hasattr(c, 'allow_gpu_growth'):
                growth = c.allow_gpu_growth
            else:
                growth = True

            config = tf.ConfigProto()
            config.gpu_options.allow_growth = growth

            # Summaries
            self.merged_summaries = tf.summary.merge_all()
            self.train_writer = tf.summary.FileWriter(osp.join(configuration.train_dir, 'summaries'), self.graph)

            # Initializing the tensor flow variables
            self.init = tf.global_variables_initializer()

            # Launch the session
            self.sess = tf.Session(config=config)
            self.sess.run(self.init)

    def encode_layers(self, scopename,  input_pcs, reuse=False ):
        c = self.configuration
        with tf.variable_scope(scopename, reuse=reuse ):
            return c.encoder(input_pcs, **c.encoder_args )

    def decode_layers(self, scopename,  input_code, reuse=False ):
        c = self.configuration
        with tf.variable_scope(scopename, reuse=reuse ):
            with tf.variable_scope('sharedDecoder', reuse=reuse) as scope:
                print(scope)
                layer5 = c.decoder( input_code, nameprefix='branch_5decoder', scope=scope, reuse=reuse, **c.decoder_args)

                x_reconstr = tf.reshape(layer5, [-1, self.n_output[0], self.n_output[1]])
                return x_reconstr

    def _create_loss(self):
        c = self.configuration

        self.loss = 0

        if c.loss == 'chamfer':
            cost_p1_p2, _, cost_p2_p1, _ = nn_distance(self.x_b1, self.gt)
            self.loss += tf.reduce_mean(cost_p1_p2) + tf.reduce_mean(cost_p2_p1) * 0.1
            cost_p1_p2, _, cost_p2_p1, _ = nn_distance(self.x_b2, self.gt)
            self.loss += tf.reduce_mean(cost_p1_p2) + tf.reduce_mean(cost_p2_p1) * 0.1
            cost_p1_p2, _, cost_p2_p1, _ = nn_distance(self.x_b3, self.gt)
            self.loss += tf.reduce_mean(cost_p1_p2) + tf.reduce_mean(cost_p2_p1) * 0.1
            cost_p1_p2, _, cost_p2_p1, _ = nn_distance(self.x_b4, self.gt)
            self.loss += tf.reduce_mean(cost_p1_p2) + tf.reduce_mean(cost_p2_p1) * 0.1
            cost_p1_p2, _, cost_p2_p1, _ = nn_distance(self.x_b5, self.gt)
            self.loss += tf.reduce_mean(cost_p1_p2) + tf.reduce_mean(cost_p2_p1)

            self.match_errors =  cost_p1_p2 + cost_p2_p1

        elif c.loss == 'emd':
            match = approx_match(self.x_b1, self.gt)
            self.loss_1 = tf.reduce_mean(match_cost(self.x_b1, self.gt, match))
            match = approx_match(self.x_b2, self.gt)
            self.loss_2 = tf.reduce_mean(match_cost(self.x_b2, self.gt, match))
            match = approx_match(self.x_b3, self.gt)
            self.loss_3 = tf.reduce_mean(match_cost(self.x_b3, self.gt, match))
            match = approx_match(self.x_b4, self.gt)
            self.loss_4 = tf.reduce_mean(match_cost(self.x_b4, self.gt, match))
            match = approx_match(self.x_b5, self.gt)
            self.loss_5 = tf.reduce_mean(match_cost(self.x_b5, self.gt, match))
            
            self.match_errors =  match_cost(self.x_b5, self.gt, match) / self.n_input[0]

            self.loss = self.loss_1 * 0.1 + self.loss_2* 0.1 + self.loss_3* 0.1 + self.loss_4* 0.1 + self.loss_5
        else:
            print("error! you must choose one!")



        reg_losses = self.graph.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)

        if c.exists_and_is_not_none('w_reg_alpha'):
            w_reg_alpha = c.w_reg_alpha
        else:
            w_reg_alpha = 1.0

        print('reg_losses:')
        print(reg_losses)

        print('w_reg_alpha = ', w_reg_alpha)

        for rl in reg_losses:
            self.loss += (w_reg_alpha * rl)

    def _setup_optimizer(self):
        c = self.configuration

        self.lr = c.learning_rate
        if hasattr(c, 'exponential_decay') and hasattr(c, 'decay_steps'):
            self.lr = tf.train.exponential_decay(c.learning_rate, self.epoch, c.decay_steps, decay_rate=0.5, staircase=True, name="learning_rate_decay")
            self.lr = tf.maximum(self.lr, 1e-5)
            tf.summary.scalar('learning_rate', self.lr)

        self.optimizer = tf.train.AdamOptimizer(learning_rate=self.lr)
        self.train_step = self.optimizer.minimize(self.loss)


    def train(self, train_data, configuration, log_file=None ):
        c = configuration
        stats = []

        if c.saver_step is not None:
            create_dir(c.train_dir)

        epoch = int(self.sess.run(self.epoch)) 
        while epoch < c.training_epochs:
            loss, duration = self._single_epoch_train(train_data, c)
            epoch = int(self.sess.run(self.increment_epoch))
            stats.append((epoch, loss, duration))

            if epoch % c.loss_display_step == 0:
                print("Epoch:", '%04d' % (epoch), 'training time (minutes)=', "{:.4f}".format(duration / 60.0), "loss=", "{:.9f}".format(loss))
                if log_file is not None:
                    log_file.write('%04d\t%.9f\t%.4f\n' % (epoch, loss, duration / 60.0))

            # Save the models checkpoint periodically.
            if c.saver_step is not None and (epoch % c.saver_step == 0 or epoch - 1 == 0):
                checkpoint_path = osp.join(c.train_dir, MODEL_SAVER_ID)
                self.saver.save(self.sess, checkpoint_path, global_step=self.epoch)

            if c.exists_and_is_not_none('summary_step') and (epoch % c.summary_step == 0 or epoch - 1 == 0):
                summary = self.sess.run(self.merged_summaries)
                self.train_writer.add_summary(summary, epoch)

        return stats


    def _single_epoch_train(self, train_data, configuration ):
        n_examples = train_data.num_examples
        epoch_loss = 0.
        batch_size = configuration.batch_size
        n_batches = int(n_examples / batch_size)
        start_time = time.time()

        # Loop over all batches
        for _ in range(n_batches):

            batch_i, _, _ = train_data.next_batch(batch_size)

            _, loss = self.partial_fit(batch_i)

            epoch_loss += loss
        epoch_loss /= n_batches
        duration = time.time() - start_time
        
        if configuration.loss == 'emd':
            epoch_loss /= len(train_data.point_clouds[0])
        
        return epoch_loss, duration

   
    def partial_fit(self, X ):
        '''Trains the model with mini-batches of input data.
        Returns:
            The loss of the mini-batch.
            The reconstructed (output) point-clouds.
        '''
        is_training(True, session=self.sess)
        try:
            _, loss, recon = self.sess.run((self.train_step, self.loss, self.x_reconstr), feed_dict={self.x: X})
            is_training(False, session=self.sess)
        except Exception:
            raise
        finally:
            is_training(False, session=self.sess)
        return recon, loss

    def reconstruct(self, X, GT=None, compute_loss=True):
        '''Use AE to reconstruct given data.
        GT will be used to measure the loss (e.g., if X is a noisy version of the GT)'''
        if compute_loss:
            loss = self.loss
        else:
            loss = self.no_op

        if GT is None:
            return self.sess.run((self.x_reconstr, loss), feed_dict={self.x: X})
        else:
            return self.sess.run((self.x_reconstr, loss), feed_dict={self.x: X, self.gt: GT})

    def transform(self, X):
        '''Transform data by mapping it into the latent space.'''
        return self.sess.run(self.z, feed_dict={self.x: X})


    def decode(self, z):
        if np.ndim(z) == 1:  # single example
            z = np.expand_dims(z, 0)
        return self.sess.run((self.x_reconstr), {self.z: z})

        
    def get_latent_codes(self, pclouds ):
        '''  wrapper of self.transform, to get the latent (bottle-neck) codes for a set of input point clouds.
        Args:
            pclouds (N, K, 3) numpy array of N point clouds with K points each.
        '''

        num2skip = len(pclouds) % self.batch_size
        idx = np.arange(len(pclouds)- num2skip)

        latent_codes = []
        for b in iterate_in_chunks(idx, self.batch_size):
            latent_codes.append(self.transform(pclouds[b]))

        # deal with remainder
        if num2skip>0:
            theRestData = pclouds[  len(pclouds)-num2skip : len(pclouds) ]
            theRestData = np.tile( theRestData,  (self.batch_size,1, 1) )
            encodeResult = self.transform(theRestData[0:self.batch_size])
            latent_codes.append( encodeResult[0:num2skip] )

        return np.vstack(latent_codes)



    def get_point_clouds(self, latentcodes ):


        num2skip = len(latentcodes) % self.batch_size
        idx = np.arange(len(latentcodes)-num2skip)

        pointclouds = []
        for b in iterate_in_chunks(idx, self.batch_size):
            pointclouds.append(self.decode(latentcodes[b]))

        # deal with remainder
        if num2skip>0:
            theRestData = latentcodes[  len(latentcodes)-num2skip : len(latentcodes) ]
            theRestData = np.tile( theRestData,  (self.batch_size,1) )
            decodeResult = self.decode( theRestData[0:self.batch_size] )
            pointclouds.append( decodeResult[0:num2skip] )

        return np.vstack(pointclouds)

