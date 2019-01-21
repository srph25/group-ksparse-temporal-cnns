import numpy as np
from scipy.misc import imresize
from keras.datasets import cifar10
from keras.utils import to_categorical
from keras_applications.imagenet_utils import preprocess_input
from datasets.mnistrotated import MNISTRotatedDataset
from utils.preprocessing import ZCAWhitening


class CIFAR10ScannedDataset(MNISTRotatedDataset):

    def __init__(self, config):
        """Constructor.
        """
        
        self.config = config
        
        (X_train, y_train), (X_test, y_test) = cifar10.load_data()
        X_train = X_train.reshape(-1, 32, 32, 3)
        X_test = X_test.reshape(-1, 32, 32, 3)
        X_train = X_train.astype(self.config['dtype'])
        X_test = X_test.astype(self.config['dtype'])
        N = ((32 - self.config['size']) // self.config['stride']) + 1
        X_train_scan = np.zeros((X_train.shape[0], N ** 2, self.config['size'], self.config['size'], 3))
        X_test_scan = np.zeros((X_test.shape[0], N ** 2, self.config['size'], self.config['size'], 3))
        for i in range(len(X_train)):
            for t1 in range(N):
                for t2 in (range(N) if (np.mod(t1, 2) == 0) else range(N - 1, -1, -1)):
                    X_train_scan[i, t1 * N + (t2 if (np.mod(t1, 2) == 0) else (N - 1 - t2)), :, :, :] = X_train[i, (t2 * self.config['stride']):(t2 * self.config['stride'] + self.config['size']), (t1 * self.config['stride']):(t1 * self.config['stride'] + self.config['size']), :]
        for i in range(len(X_test)):
            for t1 in range(N):
                for t2 in (range(N) if (np.mod(t1, 2) == 0) else range(N - 1, -1, -1)):
                    X_test_scan[i, t1 * N + (t2 if (np.mod(t1, 2) == 0) else (N - 1 - t2)), :, :, :] = X_test[i, (t2 * self.config['stride']):(t2 * self.config['stride'] + self.config['size']), (t1 * self.config['stride']):(t1 * self.config['stride'] + self.config['size']), :]
        X_train_scan /= 255.
        X_test_scan /= 255.

        # zca
        if self.config['zca_epsilon'] is not None:
            self.ZCA = ZCAWhitening(epsilon=self.config['zca_epsilon'])
            X_train_scan = self.ZCA.fit_transform(X_train_scan)
            X_test_scan = self.ZCA.transform(X_test_scan)

        if self.config['imagenet_preprocessing'] is True:
            X_train_scan = np.pad(X_train_scan, ((0, 0), (0, 0), (0, 32 - self.config['size']), (0, 32 - self.config['size']), (0, 0)), 'constant')
            X_test_scan = np.pad(X_test_scan, ((0, 0), (0, 0), (0, 32 - self.config['size']), (0, 32 - self.config['size']), (0, 0)), 'constant')
            for i in range(len(X_train_scan)):
                for t in range(len(X_train_scan[0])):
                    X_train_scan[i, t] = imresize(X_train_scan[i, t], (32, 32))
            for i in range(len(X_test_scan)):
                for t in range(len(X_test_scan[0])):
                    X_test_scan[i, t] = imresize(X_test_scan[i, t], (32, 32))
            X_train_scan = preprocess_input(X_train_scan, data_format='channels_last')
            X_test_scan = preprocess_input(X_test_scan, data_format='channels_last')
            
        nb_classes = 10
        Y_train = to_categorical(y_train, nb_classes)
        Y_test = to_categorical(y_test, nb_classes)

        self.X_train = X_train_scan
        self.Y_train = Y_train
        self.X_test = X_test_scan
        self.Y_test = Y_test
        print(self.X_train.shape, self.Y_train.shape, self.X_test.shape, self.Y_test.shape)

