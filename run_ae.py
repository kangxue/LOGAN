import sys
import os

import tensorflow as tf
import scipy.io as sio

import PIL
import argparse
import os.path as osp
import numpy as np

from config import Configuration
from AE import AutoEncoder
from in_out import  create_dir,  load_point_clouds_under_folder, output_point_cloud_ply
from latent_3d_points.tf_utils import reset_tf_graph
from general_utils import plot_3d_point_cloud_to_Image
from encoders_decoders import  decoder_with_fc_only, ocEncoder_PointNET2_multilevel256_3mlp


parser = argparse.ArgumentParser()
parser.add_argument('--gpu', default='0',  help='which gpu?')
parser.add_argument('--class_name_A', default='chair' )
parser.add_argument('--class_name_B', default='table' )

parser.add_argument('--mode', type=str, default='train', help='train or test')
parser.add_argument('--n_epochs', type=int, default=400, help='number of epochs to train')

parser.add_argument('--load_pre_trained_ae', type=int, default=0, help='0: not load pretrained AE;  1:  load pretrained AE')
parser.add_argument('--restore_epoch', type=int,  default=400,  help='which epoch do you want to load?')



FLAGS = parser.parse_args()
if FLAGS.mode == 'test'  and  FLAGS.load_pre_trained_ae==0:
    print( "Which model?")
    exit()

print( FLAGS )

os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]=FLAGS.gpu

output_dir = 'output/'          # Use to save check-points etc.
data_dir = 'data/'              # datasets


n_pc_points = 2048                # Number of points per shape
bneck_size = 256                  # Bottleneck-AE size

class_name_A = FLAGS.class_name_A
class_name_B = FLAGS.class_name_B

experiment_name = 'two_class_ae_' + class_name_A + "-" + class_name_B


datafolder = data_dir + class_name_A + '-' + class_name_B + '/'
train_dir_A = datafolder + class_name_A + '_train'
train_dir_B = datafolder + class_name_B + '_train'
test_dir_A =  datafolder + class_name_A + '_test'
test_dir_B =  datafolder+ class_name_B + '_test'


train_params = {  'ae_loss': 'emd',
                'batch_size': 32,
              'training_epochs': FLAGS.n_epochs,
              'learning_rate': 0.0005,
              'saver_step': 10,
              'saver_max_to_keep': 1000,
              'loss_display_step': 1
              }

if n_pc_points != 2048:
    raise ValueError()

##  set encoder and decoder
encoder = ocEncoder_PointNET2_multilevel256_3mlp
decoder = decoder_with_fc_only

dims_input = [n_pc_points, 3]
enc_args = {'verbose': True }
dec_args = {'layer_sizes': [256, 512, 1024, np.prod(dims_input)],
                'b_norm': False,
                'b_norm_finish': False,
                'verbose': True
                }


# create output folder                
train_dir = create_dir(osp.join(output_dir, experiment_name, "train"  ))
plot_dir  = create_dir(osp.join(output_dir, experiment_name, "plot"  ))
test_dir  = create_dir(osp.join(output_dir, experiment_name, "test"  ))


conf = Configuration(\
            n_input = [n_pc_points, 3],
            loss = train_params['ae_loss'],
            training_epochs = train_params['training_epochs'],
            batch_size = train_params['batch_size'],
            learning_rate = train_params['learning_rate'],
            train_dir = train_dir,
            loss_display_step = train_params['loss_display_step'],
            saver_step = train_params['saver_step'],
            saver_max_to_keep = train_params['saver_max_to_keep'],
            encoder = encoder,
            decoder = decoder,
            encoder_args = enc_args,
            decoder_args = dec_args,
            experiment_name = experiment_name
           )
conf.save(osp.join(train_dir, 'configuration'))

# Build AE Model.
reset_tf_graph()

ae = AutoEncoder(name=conf.experiment_name,  configuration=conf)


# load pretrained model
if FLAGS.load_pre_trained_ae:
    conf = Configuration.load(train_dir + '/configuration')
    reset_tf_graph()
    ae = AutoEncoder(conf.experiment_name, conf)
    ae.restore_model(conf.train_dir, epoch=FLAGS.restore_epoch)


batch_size =  train_params['batch_size'] 

if FLAGS.mode == 'train' :

        
    training_pc_data_A = load_point_clouds_under_folder( train_dir_A, n_threads=8, file_ending='.ply', verbose=True)
    training_pc_data_B = load_point_clouds_under_folder( train_dir_B, n_threads=8, file_ending='.ply', verbose=True)

    training_pc_data = training_pc_data_A
    training_pc_data.merge( training_pc_data_B )
    training_pc_data.shuffle_data()
    print(  'training_pc_data.point_clouds.shape[0] = ' + str(training_pc_data.point_clouds.shape[0]) )


    fout = open(osp.join(conf.train_dir, 'train_stats.txt'), 'a', 1)  # line buffering
    train_stats = ae.train(training_pc_data, conf, log_file=fout)
    fout.close()
        
elif FLAGS.mode == 'test':

    # load test set
    test_pc_data_A = load_point_clouds_under_folder( test_dir_A, n_threads=8, file_ending='.ply', verbose=True)
    test_pc_data_B = load_point_clouds_under_folder( test_dir_B, n_threads=8, file_ending='.ply', verbose=True)

    # check whether the dataset is 2D or 3D,  for plotting
    dataIs2D = False
    dataDims = np.amax( test_pc_data_A.point_clouds, axis=(0,1) ) - np.amin( test_pc_data_A.point_clouds, axis=(0,1) )
    print("dataDims = ", dataDims)
    assert(  len(dataDims.shape)==1 and  dataDims.shape[0] == 3)
    dataIs2D = any(dataDims<0.01)
    print("dataIs2D = ", dataIs2D)

    
    outshapenum = batch_size  # how many examples do you want to output?

    for X in ['A', 'B']:
        
        latentcodes = None
        errors = None

        test_dir_ply = create_dir(osp.join(test_dir, X + "_ply"))
        test_dir_png = create_dir(osp.join(test_dir, X + "_png"))

        if X=='A':
            test_pc_data = test_pc_data_A
        elif X=='B':
            test_pc_data = test_pc_data_B
        else:
            print("something is wrong...")
            exit()


        for iter in range( test_pc_data.point_clouds.shape[0] // batch_size ):

            print('batch number: ' + str( iter) )

            # Get a batch of reconstuctions and their latent-codes.
            feed_pc, feed_model_names, _ = test_pc_data.next_batch(batch_size)
            lcode, reconstructions, error \
                = ae.sess.run((ae.z, ae.x_reconstr, ae.match_errors), feed_dict={ae.x: feed_pc})

            if latentcodes is None:
                latentcodes = lcode
            else:
                latentcodes = np.concatenate((latentcodes, lcode), axis=0)

            if errors is None:
                errors = error
            else:
                errors = np.concatenate((errors, error), axis=0)

            if iter * batch_size < outshapenum:
                for i in range(batch_size):

                    output_point_cloud_ply(feed_pc[i], test_dir_ply + '/' + feed_model_names[i] + '.in.ply')
                    output_point_cloud_ply(reconstructions[i], test_dir_ply + '/' + feed_model_names[i] + '.out.ply')

                    img1 = plot_3d_point_cloud_to_Image(feed_pc[i][:, 0],  feed_pc[i][:, 1], feed_pc[i][:, 2],  dataIs2D=dataIs2D)
                    img2 = plot_3d_point_cloud_to_Image(reconstructions[i][:, 0],  reconstructions[i][:, 1], reconstructions[i][:, 2],  dataIs2D=dataIs2D)
                    img12 = PIL.Image.fromarray( np.concatenate( (img1, img2), axis=1)  )
                    img12.save(test_dir_png + '/' + feed_model_names[i]+ '.png')


        sio.savemat(test_dir +'/test_' + X + '.mat', {'latentcodes': latentcodes ,  'errors': errors  } )

else:
    print("train or test?")
