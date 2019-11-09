import tensorflow as tf
import numpy as np
import warnings

from tflearn.layers.core import fully_connected, dropout
from tflearn.layers.conv import conv_1d, avg_pool_1d
from tflearn.layers.normalization import batch_normalization
from tflearn.layers.core import fully_connected, dropout

from latent_3d_points.tf_utils import expand_scope_by_name, replicate_parameter_for_all_layers

import os
import sys
import collections

BASE_DIR = os.path.dirname(__file__)
sys.path.append(BASE_DIR)
sys.path.append(BASE_DIR + "/pointnet_plusplus/utils")
sys.path.append(BASE_DIR + "/pointnet_plusplus/tf_ops")
sys.path.append(BASE_DIR + "/pointnet_plusplus/tf_ops/3d_interpolation")
sys.path.append(BASE_DIR + "/pointnet_plusplus/tf_ops/grouping")
sys.path.append(BASE_DIR + "/pointnet_plusplus/tf_ops/sampling")
from pointnet_util import pointnet_sa_module, pointnet_fp_module



def ocEncoder_PointNET2_multilevel256_3mlp(input_points, verbose=True, is_training=None, bn_decay=None):


    l0_xyz = input_points
    l0_points = None

    # Set Abstraction layers
    l1_xyz, l1_points, l1_indices = pointnet_sa_module(l0_xyz, l0_points, npoint=512, radius=0.1, nsample=64,
                                                       mlp=[64, 64, 128], mlp2=None, group_all=False,
                                                       is_training=is_training, bn_decay=bn_decay, scope='layer1', bn=False)

    l2_xyz, l2_points, l2_indices = pointnet_sa_module(l1_xyz, l1_points, npoint=256, radius=0.2, nsample=64,
                                                       mlp=[128, 128, 256], mlp2=None, group_all=False,
                                                       is_training=is_training, bn_decay=bn_decay, scope='layer2', bn=False)

    l3_xyz, l3_points, l3_indices = pointnet_sa_module(l2_xyz, l2_points, npoint=128, radius=0.3, nsample=64,
                                                       mlp=[256, 256, 256], mlp2=None, group_all=False,
                                                       is_training=is_training, bn_decay=bn_decay, scope='layer3', bn=False)

    l4_xyz, l4_points, l4_indices = pointnet_sa_module(l3_xyz, l3_points, npoint=32, radius=0.4, nsample=64,
                                                       mlp=[256, 256, 256], mlp2=None, group_all=False,
                                                       is_training=is_training, bn_decay=bn_decay, scope='layer4', bn=False)

													   
    output_1 = encoder_with_convs_and_symmetry(l1_points, n_filters=[128, 128,  64])
    output_2 = encoder_with_convs_and_symmetry(l2_points, n_filters=[256, 256,  64])
    output_3 = encoder_with_convs_and_symmetry(l3_points, n_filters=[256, 256,  64])
    output_4 = encoder_with_convs_and_symmetry(l4_points, n_filters=[256, 256,  64])

    output_1234 = tf.concat( [output_1, output_2, output_3, output_4] , axis=1 )


    print('output_1.shape = %s', output_1.shape)
    print('output_2.shape = %s', output_2.shape)
    print('output_3.shape = %s', output_3.shape)
    print('output_4.shape = %s', output_4.shape)

    return output_1234


def encoder_with_convs_and_symmetry(in_signal, n_filters=[64, 128, 256, 1024], filter_sizes=[1], strides=[1],
                                    b_norm=True, non_linearity=tf.nn.relu, regularizer=None, weight_decay=0.001,
                                    symmetry=tf.reduce_max, dropout_prob=None, pool=avg_pool_1d, pool_sizes=None,
                                    scope=None,
                                    reuse=False, padding='same', verbose=False, closing=None, conv_op=conv_1d):


    if verbose:
        print('encoder_with_convs_and_symmetry')

    n_layers = len(n_filters)
    filter_sizes = replicate_parameter_for_all_layers(filter_sizes, n_layers)
    strides = replicate_parameter_for_all_layers(strides, n_layers)
    dropout_prob = replicate_parameter_for_all_layers(dropout_prob, n_layers)

    if n_layers < 1:
        raise ValueError('More than 0 layers are expected.')

    for i in range(n_layers):
        if i == 0:
            layer = in_signal

        name = 'encoder_conv_layer_' + str(i)
        scope_i = expand_scope_by_name(scope, name)
        layer = conv_op(layer, nb_filter=n_filters[i], filter_size=filter_sizes[i], strides=strides[i],
                        regularizer=regularizer,
                        weight_decay=weight_decay, name=name, reuse=reuse, scope=scope_i, padding=padding)

        if verbose:
            print( name, 'conv params = ', np.prod(layer.W.get_shape().as_list()) + np.prod(layer.b.get_shape().as_list()) )

        if b_norm:
            name += '_bnorm'
            scope_i = expand_scope_by_name(scope, name)
            layer = batch_normalization(layer, name=name, reuse=reuse, scope=scope_i)
            if verbose:
                print( 'bnorm params = ', np.prod(layer.beta.get_shape().as_list()) + np.prod( layer.gamma.get_shape().as_list())   )

        if non_linearity is not None:
            layer = non_linearity(layer)

        if pool is not None and pool_sizes is not None:
            if pool_sizes[i] is not None:
                layer = pool(layer, kernel_size=pool_sizes[i])

        if dropout_prob is not None and dropout_prob[i] > 0:
            layer = dropout(layer, 1.0 - dropout_prob[i])

        if verbose:
            print( layer  )
            print( 'output size:', np.prod(layer.get_shape().as_list()[1:]), '\n' )

    if symmetry is not None:
        layer = symmetry(layer, axis=1)
        if verbose:
            print
            layer

    if closing is not None:
        layer = closing(layer)
        print( layer )

    return layer



def decoder_with_fc_only(latent_signal, layer_sizes=[], b_norm=True, non_linearity=tf.nn.relu,
                         regularizer=None, weight_decay=0.001, reuse=False, scope=None, dropout_prob=None,
                         b_norm_finish=False, verbose=False, nameprefix='decoder_fc_'):
    '''A decoding network which maps points from the latent space back onto the data space.
    '''
    if verbose:
        print( 'Building Decoder' )

    n_layers = len(layer_sizes)
    dropout_prob = replicate_parameter_for_all_layers(dropout_prob, n_layers)

    if n_layers < 1:
        raise ValueError('For an FC decoder with single a layer use simpler code.')

    layer = latent_signal

    for i in range(0, n_layers - 1):
        name = nameprefix  + str(i)
        scope_i = expand_scope_by_name(scope, name)

        print('***************')
        print(scope)
        print(name)
        print(scope_i)
        print('***************')

        layer = fully_connected(layer, layer_sizes[i], activation='linear', weights_init='xavier', name=name,
                                regularizer=regularizer, weight_decay=weight_decay, reuse=reuse, scope=scope_i)

        if verbose:
            print( name, 'FC params = ', np.prod(layer.W.get_shape().as_list()) + np.prod(layer.b.get_shape().as_list()) )

        if b_norm:
            name += '_bnorm'
            scope_i = expand_scope_by_name(scope, name)
            layer = batch_normalization(layer, name=name, reuse=reuse, scope=scope_i)
            if verbose:
                print('bnorm params = ', np.prod(layer.beta.get_shape().as_list()) + np.prod( layer.gamma.get_shape().as_list())  )

        if non_linearity is not None:
            layer = non_linearity(layer)

        if dropout_prob is not None and dropout_prob[i] > 0:
            layer = dropout(layer, 1.0 - dropout_prob[i])

        if verbose:
            print( layer )
            print( 'output size:', np.prod(layer.get_shape().as_list()[1:]), '\n' )

    # Last decoding layer never has a non-linearity.
    name = nameprefix + str(n_layers - 1)
    scope_i = expand_scope_by_name(scope, name)

    print('***************')
    print(scope)
    print(name)
    print(scope_i)
    print('***************')

    layer = fully_connected(layer, layer_sizes[n_layers - 1], activation='linear', weights_init='xavier', name=name,
                            regularizer=regularizer, weight_decay=weight_decay, reuse=reuse, scope=scope_i)
    if verbose:
        print( name, 'FC params = ', np.prod(layer.W.get_shape().as_list()) + np.prod(layer.b.get_shape().as_list()) )

    if b_norm_finish:
        name += '_bnorm'
        scope_i = expand_scope_by_name(scope, name)
        layer = batch_normalization(layer, name=name, reuse=reuse, scope=scope_i)
        if verbose:
            print('bnorm params = ', np.prod(layer.beta.get_shape().as_list()) + np.prod(layer.gamma.get_shape().as_list()) )

    if verbose:
        print(  layer )
        print(  'output size:', np.prod(layer.get_shape().as_list()[1:]), '\n' )

    return layer



