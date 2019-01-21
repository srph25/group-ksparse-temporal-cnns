from keras.models import Model
from keras.layers import *
from keras.regularizers import l2
from keras import backend as K
from algorithms.kerasdenoisingcnn import KerasDenoisingCNN
from algorithms.keraswtacrnn import KerasWTACRNN
from utils.layers import *

class KerasDenoisingCRNN(KerasDenoisingCNN, KerasWTACRNN):
    def __init__(self, results_dir, config):
        super(KerasDenoisingCRNN, self).__init__(results_dir, config)

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

        enc_crnn1 = ConvMinimalRNN2D(self.config['filters'], self.config['filter_size_enc'], recurrent_activation='sigmoid', padding='same', return_sequences=True, unroll=True, kernel_regularizer=l2(self.config['l2']))
        enc_crnn1_out = enc_crnn1(enc_tdconv3_out)
        self.encoder = Model(inp, enc_crnn1_out)

