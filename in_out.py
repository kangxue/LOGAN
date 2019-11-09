import six
import warnings
import numpy as np
import os
import os.path as osp
import re
from six.moves import cPickle
from multiprocessing import Pool

import csv
from latent_3d_points.python_plyfile.plyfile import PlyElement, PlyData



def create_dir(dir_path):
    ''' Creates a directory (or nested directories) if they don't exist.
    '''
    if not osp.exists(dir_path):
        os.makedirs(dir_path)

    return dir_path

def pickle_data(file_name, *args):
    '''Using (c)Pickle to save multiple python objects in a single file.
    '''
    myFile = open(file_name, 'wb')
    cPickle.dump(len(args), myFile, protocol=2)
    for item in args:
        cPickle.dump(item, myFile, protocol=2)
    myFile.close()


def unpickle_data(file_name):
    '''Restore data previously saved with pickle_data().
    '''
    inFile = open(file_name, 'rb')
    size = cPickle.load(inFile)
    for _ in range(size):
        yield cPickle.load(inFile)
    inFile.close()


def files_in_subdirs(top_dir, search_pattern):
    regex = re.compile(search_pattern)
    for path, _, files in os.walk(top_dir):
        for name in files:
            full_name = osp.join(path, name)
            if regex.search(full_name):
                yield full_name


def load_ply(file_name, with_faces=False, with_color=False):
    ply_data = PlyData.read(file_name)
    points = ply_data['vertex']
    points = np.vstack([points['x'], points['y'], points['z']]).T
    ret_val = [points]

    if with_faces:
        faces = np.vstack(ply_data['face']['vertex_indices'])
        ret_val.append(faces)

    if with_color:
        r = np.vstack(ply_data['vertex']['red'])
        g = np.vstack(ply_data['vertex']['green'])
        b = np.vstack(ply_data['vertex']['blue'])
        color = np.hstack((r, g, b))
        ret_val.append(color)

    if len(ret_val) == 1:  # Unwrap the list
        ret_val = ret_val[0]

    return ret_val


def output_point_cloud_ply(xyz,  filepath ):

        print('write: ' + filepath)

        with open( filepath, 'w') as f:
            pn = xyz.shape[0]
            f.write('ply\n')
            f.write('format ascii 1.0\n')
            f.write('element vertex %d\n' % (pn) )
            f.write('property float x\n')
            f.write('property float y\n')
            f.write('property float z\n')
            f.write('end_header\n')
            for i in range(pn):
                f.write('%f %f %f\n' % (xyz[i][0],  xyz[i][1],  xyz[i][2]) )


def pc_loader(f_name):
    ''' loads a point-cloud saved under ShapeNet's "standar" folder scheme: 
    i.e. /syn_id/model_name.ply
    '''
    tokens = f_name.split('/')
    model_id = tokens[-1].split('.')[0]
    synet_id = tokens[-2]
    return load_ply(f_name), model_id, synet_id



def load_point_clouds_under_folder(top_dir, n_threads=20, file_ending='.ply', verbose=False):
    file_names = [f for f in files_in_subdirs(top_dir, file_ending)]

    file_names = sorted(file_names)

    if len(file_names) == 10:
        print( file_names )

    print('len(file_names) = ' + str(len(file_names)))

    loader = pc_loader

    pc = loader(file_names[0])[0]

    pclouds = np.empty([len(file_names), pc.shape[0], pc.shape[1]], dtype=np.float32)
    model_names = np.empty([len(file_names)], dtype=object)
    class_ids = np.empty([len(file_names)], dtype=object)
    pool = Pool(n_threads)

    for i, data in enumerate(pool.imap(loader, file_names)):
        pclouds[i, :, :], model_names[i], class_ids[i] = data

    pool.close()
    pool.join()

    if len(np.unique(model_names)) != len(pclouds):
        warnings.warn('Point clouds with the same model name were loaded.')

    if verbose:
        print('{0} pclouds were loaded. They belong in {1} shape-classes.'.format(len(pclouds),
                                                                                  len(np.unique(class_ids))))
    model_ids = model_names
    syn_ids = class_ids


    labels = syn_ids + '_' + model_ids

    while pclouds.shape[0] < 64:
        pclouds = np.concatenate((pclouds, pclouds), axis=0)
        labels =  np.concatenate(( labels,  labels), axis=0)


    return PointCloudDataSet(pclouds, labels=labels, init_shuffle=False)



class PointCloudDataSet(object):

    def __init__(self, point_clouds,   labels=None,  latent_codes=None,  copy=True, init_shuffle=True,  disableShuffle=False, padFor128=False ):

        self.num_examples = point_clouds.shape[0]
        self.n_points = point_clouds.shape[1]
        self.disableShuffle = disableShuffle

        if labels is not None:
            assert point_clouds.shape[0] == labels.shape[0], ('points.shape: %s labels.shape: %s' % (point_clouds.shape, labels.shape))
            if copy:
                self.labels = labels.copy()
            else:
                self.labels = labels
        else:
            self.labels = np.ones(self.num_examples, dtype=np.int8)
            

        if latent_codes is not None:
            assert point_clouds.shape[0] == latent_codes.shape[0], ('point_clouds.shape: %s latent_codes.shape: %s' % (point_clouds.shape, latent_codes.shape))
        else:
            self.latent_codes = None

        if copy:
            self.point_clouds = point_clouds.copy()
            if latent_codes is not None:
                self.latent_codes = latent_codes.copy()
        else:
            self.point_clouds = point_clouds
            if latent_codes is not None:
                self.latent_codes = latent_codes

        self.epochs_completed = 0
        self._index_in_epoch = 0
        if init_shuffle:
            self.shuffle_data()

        if padFor128:
            self.point_clouds = np.vstack((self.point_clouds, self.point_clouds[-32:] ))
            self.point_clouds = np.vstack((self.point_clouds, self.point_clouds[-32:] ))
            self.point_clouds = np.vstack((self.point_clouds, self.point_clouds[-32:] ))
            self.point_clouds = np.vstack((self.point_clouds, self.point_clouds[-32:] ))

            if self.latent_codes is not None:
                self.latent_codes = np.vstack((self.latent_codes, self.latent_codes[-32:] ))
                self.latent_codes = np.vstack((self.latent_codes, self.latent_codes[-32:] ))
                self.latent_codes = np.vstack((self.latent_codes, self.latent_codes[-32:] ))
                self.latent_codes = np.vstack((self.latent_codes, self.latent_codes[-32:] ))

            if self.labels is not None:
                labelsss = self.labels.reshape([self.num_examples, 1])
                labelsss = np.vstack((labelsss, labelsss[-32:] ))
                labelsss = np.vstack((labelsss, labelsss[-32:] ))
                labelsss = np.vstack((labelsss, labelsss[-32:] ))
                labelsss = np.vstack((labelsss, labelsss[-32:] ))
                self.labels = np.squeeze(labelsss)

            self.num_examples = self.point_clouds.shape[0]

    def shuffle_data(self, seed=None):

        if self.disableShuffle:
            return self

        if seed is not None:
            np.random.seed(seed)
        perm = np.arange(self.num_examples)
        np.random.shuffle(perm)
        self.point_clouds = self.point_clouds[perm]
        self.labels = self.labels[perm]

        if self.latent_codes is not None:
            self.latent_codes = self.latent_codes[perm]

        return self

    def next_batch(self, batch_size, seed=None):
        '''Return the next batch_size examples from this data set.
        '''
        start = self._index_in_epoch
        self._index_in_epoch += batch_size
        if self._index_in_epoch > self.num_examples:
            self.epochs_completed += 1  # Finished epoch.
            self.shuffle_data(seed)
            # Start next epoch
            start = 0
            self._index_in_epoch = batch_size
        end = self._index_in_epoch

        if self.latent_codes is not None:
            return self.point_clouds[start:end],  self.labels[start:end], self.latent_codes[start:end]
        else:
            return self.point_clouds[start:end],  self.labels[start:end], None


    def full_epoch_data(self, shuffle=True, seed=None):
        '''Returns a copy of the examples of the entire data set (i.e. an epoch's data), shuffled.
        '''
        if shuffle and seed is not None:
            np.random.seed(seed)
        perm = np.arange(self.num_examples)  # Shuffle the data.
        if shuffle:
            np.random.shuffle(perm)

        pc = self.point_clouds[perm]
        lb = self.labels[perm]

        if self.latent_codes is not None:
            lc = self.latent_codes[perm]
            return pc, lb, lc
        else:
            return pc, lb, None

    def merge(self, other_data_set):
        self._index_in_epoch = 0
        self.epochs_completed = 0
        self.point_clouds = np.vstack((self.point_clouds, other_data_set.point_clouds))

        labels_1 = self.labels.reshape([self.num_examples, 1])  # TODO = move to init.
        labels_2 = other_data_set.labels.reshape([other_data_set.num_examples, 1])
        self.labels = np.vstack((labels_1, labels_2))
        self.labels = np.squeeze(self.labels)


        if self.latent_codes is not None:
            self.latent_codes = np.vstack((self.latent_codes, other_data_set.latent_codes))

        self.num_examples = self.point_clouds.shape[0]

        return self
