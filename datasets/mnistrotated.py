import numpy as np
from scipy.misc import imresize
from PIL import Image
from keras.datasets import mnist
from keras.utils import to_categorical
from keras_applications.imagenet_utils import preprocess_input
from utils.preprocessing import ZCAWhitening


class MNISTRotatedDataset():

    def __init__(self, config):
        """Constructor.
        """
        
        self.config = config
        
        (X_train, y_train), (X_test, y_test) = mnist.load_data()
        X_train = X_train.reshape(-1, 28, 28, 1)
        X_test = X_test.reshape(-1, 28, 28, 1)
        X_train = X_train.astype(self.config['dtype'])
        X_test = X_test.astype(self.config['dtype'])
        X_train_rot = np.zeros((X_train.shape[0], self.config['rotations'], self.config['size'], self.config['size'], 1))
        X_test_rot = np.zeros((X_test.shape[0], self.config['rotations'], self.config['size'], self.config['size'], 1))
        for i in range(len(X_train)):
            img = Image.fromarray(np.reshape(X_train[i, :, :, :], (28, 28)))
            img = img.convert('L')
            phase = 360. * np.random.rand()
            for t in range(self.config['rotations']):
                _img = img.rotate(phase + t * 360. / self.config['rotations'], Image.BILINEAR).resize((28, 28), Image.BILINEAR).getdata()
                _img = np.reshape(np.array(_img, dtype=np.uint8), (28, 28, 1))[:self.config['size'], :self.config['size'], :]
                X_train_rot[i, t, :, :, :] = _img.astype(dtype=self.config['dtype'])
                
        for i in range(len(X_test)):
            img = Image.fromarray(np.reshape(X_test[i, :, :, :], (28, 28)))
            img = img.convert('L')
            phase = 360. * np.random.rand()
            for t in range(self.config['rotations']):
                _img = img.rotate(phase + t * 360. / self.config['rotations'], Image.BILINEAR).resize((28, 28), Image.BILINEAR).getdata()
                _img = np.reshape(np.array(_img, dtype=np.uint8),  (28, 28, 1))[:self.config['size'], :self.config['size'], :]
                X_test_rot[i, t, :, :, :] = _img.astype(dtype=self.config['dtype'])
        X_train_rot /= 255.
        X_test_rot /= 255.

        # zca
        if self.config['zca_epsilon'] is not None:
            self.ZCA = ZCAWhitening(epsilon=self.config['zca_epsilon'])
            X_train_rot = self.ZCA.fit_transform(X_train_rot)
            X_test_rot = self.ZCA.transform(X_test_rot)
        
        if self.config['imagenet_preprocessing'] is True:
            X_train_rot = np.pad(X_train_rot, ((0, 0), (0, 0), (0, 32 - self.config['size']), (0, 32 - self.config['size']), (0, 0)), 'constant')
            X_test_rot = np.pad(X_test_rot, ((0, 0), (0, 0), (0, 32 - self.config['size']), (0, 32 - self.config['size']), (0, 0)), 'constant')
            X_train_rot = np.repeat(X_train_rot, 3, axis=-1)
            X_test_rot = np.repeat(X_test_rot, 3, axis=-1)
            for i in range(len(X_train_rot)):
                for t in range(len(X_train_rot[0])):
                    X_train_rot[i, t] = imresize(X_train_rot[i, t], (32, 32))
            for i in range(len(X_test_rot)):
                for t in range(len(X_test_rot[0])):
                    X_test_rot[i, t] = imresize(X_test_rot[i, t], (32, 32))
            X_train_rot = preprocess_input(X_train_rot, data_format='channels_last')
            X_test_rot = preprocess_input(X_test_rot, data_format='channels_last')

        nb_classes = 10
        Y_train = to_categorical(y_train, nb_classes)
        Y_test = to_categorical(y_test, nb_classes)

        self.X_train = X_train_rot
        self.Y_train = Y_train
        self.X_test = X_test_rot
        self.Y_test = Y_test
        print(self.X_train.shape, self.Y_train.shape, self.X_test.shape, self.Y_test.shape)

    def generate_labeled_data(self, num_labeled=None):
        if num_labeled is None:
            num_labeled = round(0.9 * len(self.X_train))
        while True:
            perm = np.random.permutation(len(self.X_train))
            self.X_train_labeled = self.X_train[perm[:num_labeled]]
            self.Y_train_labeled = self.Y_train[perm[:num_labeled]]
            self.X_val_labeled = self.X_train[perm[num_labeled:round(1.1111 * num_labeled)]]
            self.Y_val_labeled = self.Y_train[perm[num_labeled:round(1.1111 * num_labeled)]]
            if np.all(np.any(self.Y_train_labeled, axis=0)) and np.all(np.any(self.Y_val_labeled, axis=0)):
                break
        print(self.X_train_labeled.shape, self.Y_train_labeled.shape, self.X_val_labeled.shape, self.Y_val_labeled.shape, self.X_test.shape, self.Y_test.shape)

