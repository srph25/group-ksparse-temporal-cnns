from keras.models import Model
from keras.layers import *
from algorithms.keraswtacnn import KerasWTACNN


class KerasRandomInitCNN(KerasWTACNN):
    def __init__(self, results_dir, config):
        super(KerasRandomInitCNN, self).__init__(results_dir, config)

    def build_sparsity(self):
        inp = Input(shape=(self.config['shape1'], self.config['shape2'], self.config['shape3'], self.config['filters']))
    
        self.sparsity = Model(inp, inp)

    def train(self, X_train, X_val=None):
        self.autoencoder_base.save_weights(self.results_dir + '/autoencoder.hdf5')
        pass

