import os
import tensorflow as tf
import scipy.io as sio
import PIL

import argparse
import os.path as osp
import numpy as np
import time
from config import Configuration
from AE import AutoEncoder
from in_out import  create_dir,  load_point_clouds_under_folder, PointCloudDataSet, output_point_cloud_ply
from latent_3d_points.tf_utils import reset_tf_graph
from general_utils import plot_3d_point_cloud_to_Image
from translator import  wgan_translator
from generators_discriminators import  latent_code_discriminator_222, latent_code_generator_2222
from latent_3d_points.neural_net import MODEL_SAVER_ID

currentfolder = os.path.basename( os.getcwd() )
print(currentfolder)


parser = argparse.ArgumentParser()
parser.add_argument('--gpu', default='0')
parser.add_argument('--class_name_A', default='chair')
parser.add_argument('--class_name_B', default='table')
parser.add_argument('--ae_epochs', type=int, default=400)
parser.add_argument('--bneck_size', type=int, default=256)
parser.add_argument('--n_pc_points', type=int, default=2048)

parser.add_argument('--mode', type=str, default='train', help='train or test')
parser.add_argument('--load_pre_trained_gan', type=int, default=0)
parser.add_argument('--restore_epoch_gan', type=int,  default=600 )


# GAN parameters
parser.add_argument('--gan_epochs', type=int, default=600)
parser.add_argument('--gan_batchsize', type=int, default=128)
parser.add_argument('--cycleLossWeight', type=float, default=20)
parser.add_argument('--featureLossWeight', type=float, default=20)
parser.add_argument('--lam', type=float, default=10)


FLAGS = parser.parse_args()

if FLAGS.mode == 'test'  and  FLAGS.load_pre_trained_gan==0:
    print( "Which model to test?")
    exit()

class_name_A = FLAGS.class_name_A
class_name_B = FLAGS.class_name_B
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = FLAGS.gpu

#########
#  Tip: training the translator on a single-GPU machine is much faster than training it on one GPU of a multi-GPU machine.
#######

# setup output folders
top_in_dir = 'data/'
top_out_dir = 'output/'
ae_configuration_AB = top_out_dir + 'two_class_ae_' + FLAGS.class_name_A + '-' + FLAGS.class_name_B + '/train/configuration'
experiment_name = 'Translation_' + class_name_A + '-' + class_name_B  

trans_out_dir = create_dir( osp.join(  top_out_dir,  experiment_name ) )
train_dir     = create_dir( osp.join(trans_out_dir, 'train'))
test_dir      = create_dir( osp.join(trans_out_dir, 'test'))
samples_dir   = create_dir( osp.join(trans_out_dir, 'samples' )  )

print(trans_out_dir)
print(train_dir)
print(test_dir)
print(samples_dir)


## Load pre-trained AE  
reset_tf_graph()
ae_conf_AB = Configuration.load(ae_configuration_AB)
print( ae_conf_AB.__str__() )

ae_AB = AutoEncoder(ae_conf_AB.experiment_name, ae_conf_AB)
ae_AB.restore_model(ae_conf_AB.train_dir, FLAGS.ae_epochs, verbose=True)

ae_A = ae_AB
ae_B = ae_AB


# data folders
datafolder = top_in_dir + class_name_A + '-' + class_name_B + '/'
train_dir_A = datafolder + class_name_A + '_train'
train_dir_B = datafolder + class_name_B + '_train'
test_dir_A = datafolder + class_name_A + '_test'
test_dir_B = datafolder + class_name_B + '_test'

## Load point-clouds 
training_pc_data_A = load_point_clouds_under_folder(train_dir_A, n_threads=8, file_ending='.ply', verbose=True)
training_pc_data_B = load_point_clouds_under_folder(train_dir_B, n_threads=8, file_ending='.ply', verbose=True)

test_pc_data_A = load_point_clouds_under_folder(test_dir_A, n_threads=8, file_ending='.ply', verbose=True)
test_pc_data_B = load_point_clouds_under_folder(test_dir_B, n_threads=8, file_ending='.ply', verbose=True)


# Use AE to convert raw pointclouds into latent codes.
data_train_A = PointCloudDataSet(point_clouds=training_pc_data_A.point_clouds, labels=training_pc_data_A.labels,  \
                                    latent_codes=ae_A.get_latent_codes(training_pc_data_A.point_clouds))
print( 'Shape of DATA train A =', data_train_A.point_clouds.shape )

data_train_B = PointCloudDataSet(point_clouds=training_pc_data_B.point_clouds, labels=training_pc_data_B.labels,  \
                                    latent_codes=ae_B.get_latent_codes(training_pc_data_B.point_clouds))
print( 'Shape of DATA train B =', data_train_B.point_clouds.shape )


# Use AE to convert raw pointclouds into latent codes.
data_test_A = PointCloudDataSet(point_clouds=test_pc_data_A.point_clouds, labels=test_pc_data_A.labels,  \
                                latent_codes=ae_A.get_latent_codes(test_pc_data_A.point_clouds) ,  init_shuffle=False)
print( 'Shape of DATA test A =', data_test_A.point_clouds.shape )

data_test_B = PointCloudDataSet(point_clouds=test_pc_data_B.point_clouds, labels=test_pc_data_B.labels,  \
                                latent_codes=ae_B.get_latent_codes(test_pc_data_B.point_clouds), init_shuffle=False)
print( 'Shape of DATA test B =', data_test_B.point_clouds.shape )

# check whether the dataset is 2D or 3D,  for plotting
dataIs2D = False
dataDims = np.amax( test_pc_data_A.point_clouds, axis=(0,1) ) - np.amin( test_pc_data_A.point_clouds, axis=(0,1) )
print("dataDims = ", dataDims)
assert(  len(dataDims.shape)==1 and  dataDims.shape[0] == 3)
dataIs2D = any(dataDims<0.01)
print("dataIs2D = ", dataIs2D)


# create a translator
init_lr = 0.002
beta = 0.5  

reset_tf_graph()
WGAN = wgan_translator(name=experiment_name, init_lr=init_lr, lam=FLAGS.lam, \
                                cycleLossWeight=FLAGS.cycleLossWeight, \
                                featureLossWeight=FLAGS.featureLossWeight, \
                                npoints=[FLAGS.n_pc_points], \
                                sizeBNeck=[FLAGS.bneck_size], \
                                discriminator=latent_code_discriminator_222, \
                                generator=latent_code_generator_2222, \
                                ae_AB=ae_AB, ae_epoch=FLAGS.ae_epochs, \
                                batch_size=FLAGS.gan_batchsize, beta=beta  )


# load pretrained model
if FLAGS.load_pre_trained_gan:
    WGAN.restore_model(train_dir, epoch=FLAGS.restore_epoch_gan)


if FLAGS.mode == 'train' :

    fout = open(osp.join(train_dir, 'train_stats.txt'), 'a', 1)  # line buffering
    saver_step = np.hstack([np.array([1, 5, 10]), np.arange(50, FLAGS.gan_epochs + 1, 50)])

    # Train the translator
    epoch = int(WGAN.sess.run(WGAN.epoch)) 
    while epoch < FLAGS.gan_epochs:

        loss, otherInfo = WGAN._single_epoch_train(data_train_A, data_train_B  )

        epoch = int(WGAN.sess.run(WGAN.increment_epoch))
        duration = otherInfo[0]
        lrate = otherInfo[1]

        print('\n')
        print(epoch, format(duration, '.4f'), lrate)
        print(  ' '.join(format(f, '.4f') for f in loss)  )

        fout.write('\n')
        fout.write( str(epoch) + ',  ' + format(duration, '.4f') + ',  ' +  str(lrate)  + '\n' )
        fout.write( ' '.join(format(f, '.4f') for f in loss)  )

        if epoch in saver_step:
            checkpoint_path = osp.join(train_dir, MODEL_SAVER_ID)
            WGAN.saver.save(WGAN.sess, checkpoint_path, global_step=WGAN.epoch)

        if epoch % 10  == 0:

            print('====== output samples  =======')
            input_pc_A, syn_pc_A2B = WGAN.translate_PointClouds(data_train_A, 'A2B', FLAGS.gan_batchsize, onlyOnebatch=True)
            input_pc_B, syn_pc_B2A = WGAN.translate_PointClouds(data_train_B, 'B2A', FLAGS.gan_batchsize, onlyOnebatch=True)

            outputNum = 1
            for k in range(outputNum):  
            
                img1 = plot_3d_point_cloud_to_Image(input_pc_A[k][:, 0],  input_pc_A[k][:, 1], input_pc_A[k][:, 2],  dataIs2D=dataIs2D)            
                img2 = plot_3d_point_cloud_to_Image(syn_pc_A2B[k][:, 0],  syn_pc_A2B[k][:, 1], syn_pc_A2B[k][:, 2],  dataIs2D=dataIs2D)
                img12 = PIL.Image.fromarray( np.concatenate( (img1, img2), axis=1)  )
                img12.save(samples_dir + '/'   + str(epoch) + '.' + str(k) + '.A2B.png' )

                img1 = plot_3d_point_cloud_to_Image(input_pc_B[k][:, 0],  input_pc_B[k][:, 1], input_pc_B[k][:, 2],  dataIs2D=dataIs2D)            
                img2 = plot_3d_point_cloud_to_Image(syn_pc_B2A[k][:, 0],  syn_pc_B2A[k][:, 1], syn_pc_B2A[k][:, 2],  dataIs2D=dataIs2D)
                img12 = PIL.Image.fromarray( np.concatenate( (img1, img2), axis=1)  )
                img12.save(samples_dir + '/'   + str(epoch) + '.' + str(k) + '.B2A.png' )
                
    fout.close()

elif FLAGS.mode == 'test':

    ## translate and save latent codes
    data_test_A_padded = PointCloudDataSet(point_clouds=data_test_A.point_clouds,   labels=data_test_A.labels, latent_codes=data_test_A.latent_codes,  init_shuffle=False, padFor128=True )
    data_test_B_padded = PointCloudDataSet(point_clouds=data_test_B.point_clouds,   labels=data_test_B.labels, latent_codes=data_test_B.latent_codes,  init_shuffle=False, padFor128=True )

    input_code_A, syn_code_A2B = WGAN.translate_code(data_test_A_padded, 'A2B', FLAGS.gan_batchsize)
    input_code_B, syn_code_B2A = WGAN.translate_code(data_test_B_padded, 'B2A', FLAGS.gan_batchsize)
    sio.savemat( test_dir + '/code_'+class_name_A + '-' + class_name_B  + '.mat', \
                                                {'input_code_A': input_code_A[ : data_test_A.num_examples ] ,  \
                                                'syn_code_A2B': syn_code_A2B[ : data_test_A.num_examples ] ,  \
                                                'input_code_B': input_code_B[ : data_test_B.num_examples ] ,  \
                                                'syn_code_B2A': syn_code_B2A[ : data_test_B.num_examples ] }   )

    # translate point clouds
    data_test_A_padded = PointCloudDataSet(point_clouds=data_test_A.point_clouds,   labels=data_test_A.labels, latent_codes=data_test_A.latent_codes,  init_shuffle=False, padFor128=True )
    data_test_B_padded = PointCloudDataSet(point_clouds=data_test_B.point_clouds,   labels=data_test_B.labels, latent_codes=data_test_B.latent_codes,  init_shuffle=False, padFor128=True )
    
    input_pc_A, syn_pc_A2B = WGAN.translate_PointClouds(data_test_A_padded, 'A2B', FLAGS.gan_batchsize)
    input_pc_B, syn_pc_B2A = WGAN.translate_PointClouds(data_test_B_padded, 'B2A', FLAGS.gan_batchsize)

    
    input_pc_A = input_pc_A[ : data_test_A.num_examples ]
    syn_pc_A2B = syn_pc_A2B[ : data_test_A.num_examples ]

    input_pc_B = input_pc_B[ : data_test_B.num_examples ]
    syn_pc_B2A = syn_pc_B2A[ : data_test_B.num_examples ]


    print(input_pc_A.shape)
    print(input_pc_B.shape)

    print(syn_pc_A2B.shape)
    print(syn_pc_B2A.shape)


    # save point clouds
    create_dir( osp.join(test_dir, 'input_A_ply/') )
    create_dir( osp.join(test_dir, 'input_B_ply/') )
    create_dir( osp.join(test_dir, 'output_A2B_ply/') )
    create_dir( osp.join(test_dir, 'output_B2A_ply/') )

    for k in range(  data_test_A.num_examples  ):
        output_point_cloud_ply(input_pc_A[k],  test_dir + '/input_A_ply/'   + data_test_A.labels[k] + '.ply' )
        output_point_cloud_ply(syn_pc_A2B[k],  test_dir + '/output_A2B_ply/'  + data_test_A.labels[k] + '.ply' )

    for k in range( data_test_B.num_examples  ):
        output_point_cloud_ply(input_pc_B[k],  test_dir + '/input_B_ply/'   +  data_test_B.labels[k]  + '.ply' )
        output_point_cloud_ply(syn_pc_B2A[k],  test_dir + '/output_B2A_ply/' + data_test_B.labels[k] + '.ply' )



    # Plot point clouds

    create_dir( osp.join(test_dir, 'A2B_png/') )
    create_dir( osp.join(test_dir, 'B2A_png/') )

    for k in range(data_test_A.num_examples):

        print('save A png: {}\n'.format(k) )

        img1 = plot_3d_point_cloud_to_Image(input_pc_A[k][:, 0],  input_pc_A[k][:, 1], input_pc_A[k][:, 2],  dataIs2D=dataIs2D)
        img2 = plot_3d_point_cloud_to_Image(syn_pc_A2B[k][:, 0],  syn_pc_A2B[k][:, 1], syn_pc_A2B[k][:, 2],  dataIs2D=dataIs2D)
        img12 = PIL.Image.fromarray( np.concatenate( (img1, img2), axis=1)  )
        img12.save( test_dir + '/A2B_png/' + data_test_A.labels[k] + '.png' )

    for k in range(data_test_B.num_examples):

        print('save B png: {}\n'.format(k) )

        img1 = plot_3d_point_cloud_to_Image(input_pc_B[k][:, 0],  input_pc_B[k][:, 1], input_pc_B[k][:, 2],  dataIs2D=dataIs2D)
        img2 = plot_3d_point_cloud_to_Image(syn_pc_B2A[k][:, 0],  syn_pc_B2A[k][:, 1], syn_pc_B2A[k][:, 2],  dataIs2D=dataIs2D)
        img12 = PIL.Image.fromarray( np.concatenate( (img1, img2), axis=1)  )
        img12.save( test_dir + '/B2A_png/' + data_test_B.labels[k] + '.png' )
        
    
else:
    print("train or test?")





