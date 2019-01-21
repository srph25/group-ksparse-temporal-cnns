from keras.models import Model
from keras.layers import *
from keras.regularizers import l2
from keras.applications.vgg19 import VGG19, preprocess_input, decode_predictions
from algorithms.kerasrandominitcnn import KerasRandomInitCNN


class KerasVGG19(KerasRandomInitCNN):
    def __init__(self, results_dir, config):
        super(KerasVGG19, self).__init__(results_dir, config)

    def build_encoder(self):
        inp = Input(shape=(self.config['shape1'], self.config['shape2'], self.config['shape3'], self.config['shape4']))

        vgg_model = VGG19(include_top=False, weights='imagenet', input_shape=(32, 32, 3))
        
        # freeze pre-trained model area's layer
        for layer in vgg_model.layers[:-2]:
            layer.trainable = False
        for layer in vgg_model.layers[-2:]:
            layer.trainable = True

        enc_tdvgg = TimeDistributed(vgg_model)
        enc_tdvgg_out = enc_tdvgg(inp)

        self.encoder = Model(inp, enc_tdvgg_out)

    def build_sparsity(self):
        inp = Input(shape=(self.config['shape1'], 1, 1, 512))
    
        self.sparsity = Model(inp, inp)

    def build_decoder(self):
        inp = Input(shape=(self.config['shape1'], 1, 1, 512))

        self.decoder = Model(inp, inp)

    def build_classifier(self):
        inp = Input(shape=(self.config['shape1'], 1, 1, 512))

        clf_flat = Flatten()
        clf_tdflat = TimeDistributed(clf_flat)
        clf_tdflat_out = clf_tdflat(inp)
        clf_drop = Dropout(self.config['dropout'])
        clf_tddrop = TimeDistributed(clf_drop)
        clf_tddrop_out = clf_tddrop(clf_tdflat_out)
        clf_fc = Dense(self.config['classes'], kernel_regularizer=l2(self.config['l2']))
        clf_tdfc = TimeDistributed(clf_fc)
        clf_tdfc_out = clf_tdfc(clf_tddrop_out)
        clf_collapse = Lambda(lambda x: K.mean(x, axis=1))
        clf_collapse_out = clf_collapse(clf_tdfc_out)
        clf_sm = Activation('softmax')
        clf_sm_out = clf_sm(clf_collapse_out)
        
        self.classifier = Model(inp, clf_sm_out)

    def build(self):
        super(KerasVGG19, self).build()
        self.autoencoder_base.save_weights = lambda filepath, overwrite=True: self.encoder.save_weights(filepath, overwrite=overwrite)
        self.autoencoder_base.load_weights = lambda filepath: self.encoder.load_weights(filepath)
        def save(filepath, overwrite=True):
            self.encoder.save_weights(filepath[:-4] + '_encoder.hdf5', overwrite=overwrite)
            self.classifier.save_weights(filepath[:-4] + '_classifier.hdf5', overwrite=overwrite)
        def load(filepath):
            self.encoder.load_weights(filepath[:-4] + '_encoder.hdf5')
            self.classifier.load_weights(filepath[:-4] + '_classifier.hdf5')
        self.endtoend_classifier_base.save_weights = save
        self.endtoend_classifier_base.load_weights = load
        self.stacked_classifier_base.save_weights = save
        self.stacked_classifier_base.load_weights = load

    def test(self, X):
        pass
            
    def plot_predictions(self, X, filepath=None, title='Predictions'):
        pass

