import numpy as np
from utils.pil import fromimage, toimage, imresize, imread, imsave
from keras.utils import to_categorical
from keras_applications.imagenet_utils import preprocess_input
from datasets.mnistrotated import MNISTRotatedDataset
from utils.preprocessing import ZCAWhitening


class COIL100Dataset(MNISTRotatedDataset):

    def __init__(self, config):
        """Constructor.
        """
        
        self.config = config
        
        X = np.zeros((100, 72, self.config['size'], self.config['size'], 3), dtype=self.config['dtype'])
        y = np.zeros((100,), dtype='int32')
        for i in range(len(X)):
            for j, t in enumerate(range(0, 360, 5)):
                img = imread(self.config['path'] + '/obj' + str(i + 1) + '__' + str(t) + '.png')
                img = imresize(img, (self.config['size'], self.config['size']))
                X[i, j, :, :, :] = img
            y[i] = i
        X /= 255.

        # zca
        if self.config['zca_epsilon'] is not None:
            self.ZCA = ZCAWhitening(epsilon=self.config['zca_epsilon'])
            X = self.ZCA.fit_transform(X)

        if self.config['imagenet_preprocessing'] is True:
            X = np.pad(X, ((0, 0), (0, 0), (0, 32 - self.config['size']), (0, 32 - self.config['size']), (0, 0)), 'constant')
            for i in range(len(X)):
                for t in range(len(X[0])):
                    X[i, t] = imresize(X[i, t], (32, 32))
            X = preprocess_input(X, data_format='channels_last')
            
        Y = to_categorical(y, len(X))
            
        self.X = X
        self.Y = Y
        print(self.X.shape, self.Y.shape)

    def generate_labeled_data(self, num_labeled):
        if num_labeled is None:
            num_labeled = self.X_train[1]
        ind_train = list(range(0, 64 - np.mod(64, 64 // num_labeled), 64 // num_labeled))
        ind_val = list(set(range((64 // num_labeled) // 2, 64 - np.mod(64, 64 // num_labeled), 
                                 64 // num_labeled)) - set(ind_train))
        ind_test = list(set(range(self.X.shape[1])) - set(ind_train) - set(ind_val))
        self.X_train_labeled = self.X[:, ind_train]
        self.Y_train_labeled = self.Y
        self.X_val_labeled = self.X[:, ind_val]
        self.Y_val_labeled = self.Y
        self.X_test = self.X[:, ind_test]
        self.Y_test = self.Y
        print(self.X_train_labeled.shape, self.Y_train_labeled.shape, self.X_val_labeled.shape, self.Y_val_labeled.shape, self.X_test.shape, self.Y_test.shape)

