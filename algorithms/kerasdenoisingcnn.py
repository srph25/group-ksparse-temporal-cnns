from keras.models import Model
from keras.layers import *
from keras.regularizers import l2
from keras import backend as K
from algorithms.keraswtacnn import KerasWTACNN
from utils.layers import *


class KerasDenoisingCNN(KerasWTACNN):
    def __init__(self, results_dir, config):
        super(KerasDenoisingCNN, self).__init__(results_dir, config)

    def build_encoder(self):
        inp = Input(shape=(self.config['shape1'], self.config['shape2'], self.config['shape3'], self.config['shape4']))
        
        enc_drop1 = Dropout(self.config['dropout'])(inp)
        
        enc_conv1 = Conv2D(self.config['filters'], self.config['filter_size_enc'], activation='relu', padding='same', kernel_regularizer=l2(self.config['l2']))
        enc_tdconv1 = TimeDistributed(enc_conv1)
        enc_tdconv1_out = enc_tdconv1(enc_drop1)

        enc_conv2 = Conv2D(self.config['filters'], self.config['filter_size_enc'], activation='relu', padding='same', kernel_regularizer=l2(self.config['l2']))
        enc_tdconv2 = TimeDistributed(enc_conv2)
        enc_tdconv2_out = enc_tdconv2(enc_tdconv1_out)

        enc_conv3 = Conv2D(self.config['filters'], self.config['filter_size_enc'], activation='relu', padding='same', kernel_regularizer=l2(self.config['l2']))
        enc_tdconv3 = TimeDistributed(enc_conv3)
        enc_tdconv3_out = enc_tdconv3(enc_tdconv2_out)

        self.encoder = Model(inp, enc_tdconv3_out)

    def build_sparsity(self):
        inp = Input(shape=(self.config['shape1'], self.config['shape2'], self.config['shape3'], self.config['filters']))

        self.sparsity = Model(inp, inp)

