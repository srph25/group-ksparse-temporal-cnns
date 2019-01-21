import numpy as np
from scipy.misc import imresize, imread
from keras.utils import to_categorical
from keras_applications.imagenet_utils import preprocess_input
from datasets.coil100 import COIL100Dataset
from utils.preprocessing import ZCAWhitening

class NECAnimalDataset(COIL100Dataset):

    def __init__(self, config):
        """Constructor.
        """
        
        self.config = config
        
        idxs = []
        with open(self.config['path'] + '/seq_ref_clean.txt', 'r', encoding="latin-1") as f:
            for l, line in enumerate(f):
                if l >= 4:
                    idxs.append([int(j) for j in line.split(' ')])
        idxs[-1][1] = 4371
        X = self.config['pad_value'] * np.ones((len(idxs), np.max(np.diff(np.array(idxs), axis=1)) + 1, self.config['size'], self.config['size'], 3), dtype=self.config['dtype'])
        y = np.zeros((len(idxs),), dtype='int32')
        
        for i, idx in enumerate(idxs):
            for j, q in enumerate(range(idx[0], idx[1] + 1)):
                img = imread(self.config['path'] + '/img_clean/img' + '{:04d}'.format(q) + '.jpg')
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

