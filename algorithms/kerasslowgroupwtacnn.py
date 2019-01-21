from keras.models import Model
from keras.layers import *
from keras import backend as K
from algorithms.kerasgroupwtacnn import KerasGroupWTACNN
from utils.layers import *
from utils.ops import *


class KerasSlowGroupWTACNN(KerasGroupWTACNN):
    def __init__(self, results_dir, config):
        super(KerasSlowGroupWTACNN, self).__init__(results_dir, config)

    def build_sparsity(self):
        inp = Input(shape=(self.config['shape1'], self.config['shape2'], self.config['shape3'], self.config['filters']))
        
        def slow_reg(x, lmbd):
            gn = group_norms(inputs=x, groups=self.config['groups'], axis=-1)
            return lmbd * K.mean(K.mean(K.abs(gn[:, 1:] - gn[:, :-1]), axis=-1))
        
        sparse = RegularizedLambda(lambda x: group_ksparse(group_ksparse(x, groups=self.config['groups'], k=self.config['k_spatial'], axis_group=[4], axis_sparse=[2, 3]),
                                                           groups=self.config['groups'], k=self.config['k_lifetime'], axis_group=[4], axis_sparse=[0, 1]), 
                                                           activity_regularizer=lambda x: slow_reg(x, self.config['l1_slow']))
        sparse_out = sparse(inp)

        self.sparsity = Model(inp, sparse_out)

