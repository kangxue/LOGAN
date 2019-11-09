import numpy as np
import tensorflow as tf
from encoders_decoders import  decoder_with_fc_only


########################## generator ####################

bnorm_default = True
dropout_default = None # [0.1]

def latent_code_generator_2222(z, out_dim, layer_sizes=[256,256,256, 256 ], b_norm=False, non_linearity=tf.nn.relu, reuse=False, scope=None):
    layer_sizes = layer_sizes + out_dim
    out_signal = decoder_with_fc_only(z, layer_sizes=layer_sizes, b_norm=bnorm_default, dropout_prob=dropout_default, reuse=reuse, scope=scope)
    return out_signal



#################  Discriminator ############################

bnorm_default = True
dropout_default = [0.1]

def latent_code_discriminator_222(in_signal, layer_sizes=[256, 256, 256], b_norm=False, non_linearity=tf.nn.relu, reuse=False, scope=None):
    layer_sizes = layer_sizes + [1]
    d_logit = decoder_with_fc_only(in_signal, layer_sizes=layer_sizes, non_linearity=non_linearity, b_norm=bnorm_default, dropout_prob=dropout_default, reuse=reuse, scope=scope)
    d_prob = tf.nn.sigmoid(d_logit)
    return d_prob, d_logit
