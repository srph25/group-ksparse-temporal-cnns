import os
import tensorflow as tf
from sklearn.utils.class_weight import compute_class_weight
from keras.models import Model
from keras.layers import *
from keras.optimizers import *
from keras.objectives import mean_squared_error
from keras.regularizers import l2
from keras.callbacks import TerminateOnNaN, ModelCheckpoint, EarlyStopping
from keras import backend as K
from keras.utils import multi_gpu_model, Sequence
from utils.ops import *
from utils.plot import *
from utils.utils import *


class KerasWTACNN():
    def __init__(self, results_dir, config):
        self.results_dir = results_dir
        self.set_config(config)
        self.loss = lambda true, pred: K.mean(K.square(pred - true), axis=-1)
    
    def set_config(self, config):
        assert(np.mod(config['batch_size'], config['gpus']) == 0)
        self.config = config

    def build_encoder(self):
        inp = Input(shape=(self.config['shape1'], self.config['shape2'], self.config['shape3'], self.config['shape4']))
        
        enc_conv1 = Conv2D(self.config['filters'], self.config['filter_size_enc'], activation='relu', padding='same', kernel_regularizer=l2(self.config['l2']))
        enc_tdconv1 = TimeDistributed(enc_conv1)
        enc_tdconv1_out = enc_tdconv1(inp)

        enc_conv2 = Conv2D(self.config['filters'], self.config['filter_size_enc'], activation='relu', padding='same', kernel_regularizer=l2(self.config['l2']))
        enc_tdconv2 = TimeDistributed(enc_conv2)
        enc_tdconv2_out = enc_tdconv2(enc_tdconv1_out)

        enc_conv3 = Conv2D(self.config['filters'], self.config['filter_size_enc'], activation='relu', padding='same', kernel_regularizer=l2(self.config['l2']))
        enc_tdconv3 = TimeDistributed(enc_conv3)
        enc_tdconv3_out = enc_tdconv3(enc_tdconv2_out)

        self.encoder = Model(inp, enc_tdconv3_out)

    def build_sparsity(self):
        inp = Input(shape=(self.config['shape1'], self.config['shape2'], self.config['shape3'], self.config['filters']))

        sparse = Lambda(lambda x: ksparse(ksparse(x, k=self.config['k_spatial'], axis=[2, 3], absolute=False), k=self.config['k_lifetime'], axis=[0, 1], absolute=False))
        sparse_out = sparse(inp)

        self.sparsity = Model(inp, sparse_out)

    def build_decoder(self):
        inp = Input(shape=(self.config['shape1'], self.config['shape2'], self.config['shape3'], self.config['filters']))

        dec_conv1 = Conv2DTranspose(self.config['shape4'], self.config['filter_size_dec'], activation='linear', padding='same', kernel_regularizer=l2(self.config['l2']))
        dec_tdconv1 = TimeDistributed(dec_conv1)
        dec_tdconv1_out = dec_tdconv1(inp)
        
        self.decoder = Model(inp, dec_tdconv1_out)

    def build_classifier(self):
        inp = Input(shape=(self.config['shape1'], self.config['shape2'], self.config['shape3'], self.config['filters']))

        clf_pool = MaxPooling2D(self.config['pool_size'])
        clf_tdpool = TimeDistributed(clf_pool)
        clf_tdpool_out = clf_tdpool(inp)
        clf_flat = Flatten()
        clf_tdflat = TimeDistributed(clf_flat)
        clf_tdflat_out = clf_tdflat(clf_tdpool_out)
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

    def compile(self, inputs, outputs, optimizer=None, loss=None, metrics=None, loss_weights=None):
        if self.config['gpus'] > 1:
            with tf.device('/cpu:0'):
                model_base = Model(inputs, outputs)
            model = multi_gpu_model(model_base, gpus=self.config['gpus'])
        else:
            model_base = Model(inputs, outputs)
            model = model_base
        model_base.summary()
        if optimizer is None:
            optimizer = Adam(amsgrad=False, lr=self.config['lr'], clipnorm=self.config['clipnorm'])
        if loss is None:
            loss = 'mse'
        if metrics is None:
            metrics = []
        '''
        if loss_weights is None:
            loss_weights = 1.
        '''
        model.compile(optimizer=optimizer,
                      loss=loss,
                      metrics=metrics,
                      loss_weights=loss_weights)
        return model, model_base

    def build(self):
        K.clear_session()

        self.build_encoder()
        self.build_sparsity()
        self.build_decoder()
        self.build_classifier()
        
        inp = Input(shape=(self.config['shape1'], self.config['shape2'], self.config['shape3'], self.config['shape4']))
        enc_out = self.encoder(inp)
        sparse_out = self.sparsity(enc_out)
        dec_out = self.decoder(sparse_out)
        self.autoencoder, self.autoencoder_base = self.compile(inp, dec_out, loss=self.loss)

        endtoend_clf_out = self.classifier(enc_out)
        self.endtoend_classifier, self.endtoend_classifier_base = self.compile(inp, endtoend_clf_out, 
                                                                               optimizer=SGD(lr=self.config['lr'] / 10,
                                                                                             momentum=self.config['momentum'],
                                                                                             clipnorm=self.config['clipnorm']),
                                                                               loss='categorical_crossentropy',
                                                                               metrics=['accuracy'])

        self.encoder.trainable = False
        stacked_clf_out = self.classifier(enc_out)
        self.stacked_classifier, self.stacked_classifier_base = self.compile(inp, stacked_clf_out,
                                                                             loss='categorical_crossentropy',
                                                                             metrics=['accuracy'])

    def train(self, X_train, X_val=None):
        assert(isinstance(X_train, np.ndarray) or isinstance(X_train, Sequence))
        if X_val is not None:
            assert(isinstance(X_val, np.ndarray) or isinstance(X_val, Sequence))
        # Fit!
        hdf5 = self.results_dir + '/autoencoder.hdf5'
        if os.path.exists(hdf5):
            os.remove(hdf5)
        model_checkpoint = ModelCheckpoint(filepath=hdf5, save_best_only=True if X_val is not None else False, save_weights_only=True)
        early_stopping = EarlyStopping(patience=self.config['patience'])
        terminate_on_nan = TerminateOnNaN()
        if isinstance(X_train, np.ndarray):
            self.autoencoder.fit(X_train, X_train,
                                 batch_size=self.config['batch_size'],
                                 epochs=self.config['epochs'],
                                 verbose=1,
                                 callbacks=[model_checkpoint, early_stopping, terminate_on_nan] if X_val is not None else [model_checkpoint, terminate_on_nan],
                                 validation_data=(X_val, X_val) if X_val is not None else None)
        elif isinstance(X_train, Sequence):
            self.autoencoder.fit_generator(X_train,
                                           epochs=self.config['epochs'],
                                           verbose=1,
                                           callbacks=[model_checkpoint, early_stopping, terminate_on_nan] if X_val is not None else [model_checkpoint, terminate_on_nan],
                                           validation_data=X_val if X_val is not None else None,
                                           shuffle=True, workers=14)
        self.autoencoder.load_weights(hdf5)
        self.autoencoder_base.save_weights(hdf5)
        self.plot_decoder_weights()

    def train_classifier(self, X_train, Y_train=None, X_val=None, Y_val=None, stacked=True):
        assert((isinstance(X_train, np.ndarray) and isinstance(Y_train, np.ndarray)) or isinstance(X_train, Sequence))
        if X_val is not None:
            assert((isinstance(X_val, np.ndarray) and isinstance(Y_val, np.ndarray)) or isinstance(X_val, Sequence))
        hdf5 = self.results_dir + '/supervised_' + str(stacked) + '.hdf5'
        if os.path.exists(hdf5):
            os.remove(hdf5)
        model_checkpoint = ModelCheckpoint(filepath=hdf5, save_best_only=True if (X_val is not None and Y_val is not None) else False, save_weights_only=True)
        early_stopping = EarlyStopping(patience=self.config['patience'])
        terminate_on_nan = TerminateOnNaN()
        if stacked is True:
            model = self.stacked_classifier
            model_base = self.stacked_classifier_base
        else:
            model = self.endtoend_classifier
            model_base = self.endtoend_classifier_base
        if isinstance(X_train, np.ndarray) and isinstance(Y_train, np.ndarray):
            model.fit(X_train, Y_train,
                      batch_size=self.config['batch_size'],
                      epochs=self.config['epochs'],
                      verbose=1,
                      callbacks=[model_checkpoint, early_stopping, terminate_on_nan] if (X_val is not None and Y_val is not None) else [model_checkpoint, terminate_on_nan],
                      validation_data=(X_val, Y_val) if (X_val is not None and Y_val is not None) else None,
                      class_weight=compute_class_weight('balanced', np.unique(np.argmax(Y_train, axis=-1)), np.argmax(Y_train, axis=-1)))
        elif isinstance(X_train, Sequence):
            model.fit_generator(X_train,
                                epochs=self.config['epochs'],
                                verbose=1,
                                callbacks=[model_checkpoint, early_stopping, terminate_on_nan] if (X_val is not None and Y_val is not None) else [model_checkpoint, terminate_on_nan],
                                validation_data=X_val if X_val is not None else None,
                                class_weight=compute_class_weight('balanced', np.unique(np.argmax(Y_train, axis=-1)), np.argmax(Y_train, axis=-1)) if Y_train is not None else None,
                                shuffle=True, workers=14)
        model.load_weights(hdf5)
        model_base.save_weights(hdf5)
            
    def test(self, X):
        if isinstance(X, np.ndarray):
            return self.autoencoder.evaluate(X, X, batch_size=self.config['batch_size'])
        elif isinstance(X, Sequence):
            return self.autoencoder.evaluate_generator(X, workers=14)
    
    def test_classifier(self, X, Y=None, stacked=True):
        if stacked is True:
            model = self.stacked_classifier
        else:
            model = self.endtoend_classifier
        if isinstance(X, np.ndarray) and isinstance(Y, np.ndarray):
            return model.evaluate(X, Y, batch_size=self.config['batch_size'])
        elif isinstance(X, Sequence):
            return model.evaluate_generator(X, workers=14)

    def predict(self, X):
        if isinstance(X, np.ndarray):
            return self.autoencoder.predict(X, batch_size=self.config['batch_size'])
        elif isinstance(X, Sequence):
            return self.autoencoder.predict_generator(X, workers=14)
        
    def predict_classifier(self, X, stacked=True):
        if stacked is True:
            model = self.stacked_classifier
        else:
            model = self.endtoend_classifier
        if isinstance(X, np.ndarray):
            return model.predict(X, batch_size=self.config['batch_size'])
        elif isinstance(X, Sequence):
            return model.predict_generator(X, workers=14)

    def plot_weights(self, weights, filepath=None, title='Weights'):
        if filepath is None:
            filepath = self.results_dir + '/weights.png'
        plot(filepath, title, make_mosaic(weights, nrows=4, ncols=self.config['filters'] // 4))
        
    def plot_decoder_weights(self):
        self.plot_weights(np.transpose(self.autoencoder_base.get_weights()[-2], (3, 0, 1, 2)), filepath=self.results_dir + '/decoder_weights.png', title='Decoder Weights') # -4

    def plot_predictions(self, X, filepath=None, title='Predictions'):
        if filepath is None:
            filepath = self.results_dir + '/predictions.png'
        pred = self.predict(X)
        images = np.concatenate([np.transpose(X, (1, 0, 2, 3, 4)), np.transpose(pred, (1, 0, 2, 3, 4))], axis=0)
        images = np.reshape(images, (2 * X.shape[0] * X.shape[1], X.shape[2], X.shape[3], X.shape[4]))
        plot(filepath, title, make_mosaic(images, nrows=X.shape[0], ncols=int(2 * X.shape[1]), clip=True))

