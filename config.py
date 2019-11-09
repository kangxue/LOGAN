
import warnings
import os.path as osp
import tensorflow as tf
import numpy as np

from in_out import create_dir, pickle_data, unpickle_data

class Configuration():
    def __init__(self, n_input, encoder, decoder, encoder_args={}, decoder_args={},
                 training_epochs=400, batch_size=32, learning_rate=0.0005, 
                 saver_step=None, train_dir=None,  loss='emd',   experiment_name='ae',
                 saver_max_to_keep=None, loss_display_step=1, debug=False,
                 n_output=None ):

        # Parameters for any AE
        self.n_input = n_input
        self.loss = loss.lower()
        self.decoder = decoder
        self.encoder = encoder
        self.encoder_args = encoder_args
        self.decoder_args = decoder_args

        # Training related parameters
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.loss_display_step = loss_display_step
        self.saver_step = saver_step
        self.train_dir = train_dir
        self.saver_max_to_keep = saver_max_to_keep
        self.training_epochs = training_epochs
        self.debug = debug
        self.experiment_name = experiment_name

        # Used in AP
        if n_output is None:
            self.n_output = n_input
        else:
            self.n_output = n_output


    def exists_and_is_not_none(self, attribute):
        return hasattr(self, attribute) and getattr(self, attribute) is not None

    def __str__(self):
        keys = list( self.__dict__.keys()    )
        vals = list( self.__dict__.values()  )
        index = np.argsort(keys)
        res = ''
        for i in index:
            if callable(vals[i]):
                v = vals[i].__name__
            else:
                v = str(vals[i])
            res += '%30s: %s\n' % (str(keys[i]), v)
        return res

    def save(self, file_name):
        pickle_data(file_name + '.pickle', self)
        with open(file_name + '.txt', 'w') as fout:
            fout.write(self.__str__())

    @staticmethod
    def load(file_name):
        return unpickle_data(file_name + '.pickle').__next__()
