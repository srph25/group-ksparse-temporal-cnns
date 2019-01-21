from keras.models import Model
from keras.layers import *
from algorithms.keraswtacnn import KerasWTACNN
from utils.ops import *
from utils.plot import *


class KerasGroupWTACNN(KerasWTACNN):
    def __init__(self, results_dir, config):
        super(KerasGroupWTACNN, self).__init__(results_dir, config)

    def build_sparsity(self):
        inp = Input(shape=(self.config['shape1'], self.config['shape2'], self.config['shape3'], self.config['filters']))
    
        sparse = Lambda(lambda x: group_ksparse(group_ksparse(x, groups=self.config['groups'], k=self.config['k_spatial'], axis_group=[4], axis_sparse=[2, 3]), 
                                                groups=self.config['groups'], k=self.config['k_lifetime'], axis_group=[4], axis_sparse=[0, 1]))
        sparse_out = sparse(inp)

        self.sparsity = Model(inp, sparse_out)

    def plot_weights(self, weights, filepath=None, title='Weights'):
        if filepath is None:
            filepath = self.results_dir + '/weights.png'
        plot(filepath, title, make_mosaic(weights, nrows=int(self.config['filters'] // self.config['groups']), ncols=self.config['groups']))

