import numpy as np
from scipy.misc import imresize, imread
import os
import h5py
import tqdm
from keras.utils import Sequence, to_categorical
from keras_applications.imagenet_utils import preprocess_input
from utils.preprocessing import IncrementalZCAWhitening


class VideoSequence(Sequence):
    def __init__(self, config, train_val_test, videos, grayscale=False, zca=None, classes=None):
        self.config = config
        self.train_val_test = train_val_test
        self.videos = videos
        self.get_metas()
        self.grayscale = grayscale
        self.classes = classes
        self.precomputing = False
        if self.config['zca_epsilon'] is not None:
            if zca is None:
                self.get_zca()
            else:
                self.zca = zca
        else:
            self.zca = None
            
    def __len__(self):
        # Get steps per epoch.
       return int(np.ceil(len(self.videos) / self.config['batch_size']))

    def __getitem__(self, idx):
        batch_start = idx * self.config['batch_size']
        batch_end = np.min([batch_start + self.config['batch_size'], len(self.videos)])
        metas = self.metas[batch_start:batch_end]
        lens = [v[0] for v in metas]
        X = []
        for v, video in enumerate(self.videos[batch_start:batch_end]):
            frame_start = np.random.randint(lens[v] - self.config['frames'])
            frame_end = np.min([frame_start + self.config['frames'], lens[v]])
            frames = self.build_frames(video, metas[v], frame_start=frame_start, frame_end=frame_end)
            X.append(frames)
        X = np.array(X).astype(self.config['dtype'])
        if self.config['zca_epsilon'] is not None:
            if self.precomputing is True:
                X = self.zca.partial_fit_transform(X)
            else:
                if self.train_val_test in ['train', 'train_labeled'] and 'size_crop' in self.config.keys():
                    X = self.zca.partial_fit_transform(X)
                else:
                    X = self.zca.transform(X)
        if self.config['imagenet_preprocessing'] is True:
            if X.shape[-1] == 1:
                X = np.repeat(X, 3, axis=-1)
            X = np.pad(X, ((0, 0), (0, 0), (0, 32 - self.config['size']), (0, 32 - self.config['size']), (0, 0)), 'constant')
            for i in range(len(X)):
                for t in range(len(X[0])):
                    X[i, t] = imresize(X[i, t], (32, 32))            
            X = preprocess_input(X, data_format='channels_last')
        if self.classes is None:
            Y = X
        else:
            Y = []
            for v, video in enumerate(self.videos[batch_start:batch_end]):
                y = self.classes.index(video.split('/')[0])
                Y.append(y)
            Y = to_categorical(Y, len(self.classes))
        return (X, Y)

    def get_meta(self, video):
        frames = self.get_frames(video)
        try: 
            img = imread(self.config['path'] + '/' + video.split('/')[0] + '/' + frames[0])
        except:
            img = imread(self.config['path'] + '/' + video + '/' + frames[0])
        frame_count, height, width = len(frames), img.shape[0], img.shape[1]
        return frame_count, height, width

    def get_metas(self):
        self.metas = []
        filename = self.config['path_split'] + '/' + self.train_val_test + 'list0' + str(self.config['split']) + '_' + str(self.config['frames']) + '_' + str(self.config['size']) + '_meta.txt'
        if os.path.exists(filename):
            with open(filename, 'r') as f:
                data = f.readlines()
            self.videos = [d.split(' ')[0] for d in data]
            self.metas = [[int(d.split(' ')[1]), int(d.split(' ')[2]), int(d.split(' ')[3].split('\n')[0])] for d in data]
        else:
            with open(filename, 'w') as f:
                for video in tqdm.tqdm(self.videos):
                    meta = self.get_meta(video)
                    self.metas.append(list(meta))
                    print(video, meta)
                    f.write(video + ' ' + str(meta[0]) + ' ' + str(meta[1]) + ' ' + str(meta[2]) + '\n') 

    def get_zca(self):
        self.zca = IncrementalZCAWhitening(epsilon=self.config['zca_epsilon'])
        if 'size_crop' not in self.config.keys():
            filename = self.config['path_split'] + '/' + self.train_val_test + 'list0' + str(self.config['split']) + '_' + str(self.config['frames']) + '_' + str(self.config['size']) \
                       + '_' + str(self.config['zca_epsilon']) + '_' + str(self.config['zca_epochs'])+ '_zca.hdf5'
            if os.path.exists(filename):
                print('Loading precomputed ZCA...')
                with h5py.File(filename, 'r') as f:
                    self.zca.components = f['components'][:]
                    self.zca.singular_values = f['singular_values'][:]
                    self.zca.mean = f['mean'][:]
                    self.zca.var = f['var'][:]
                    self.zca.explained_variance = f['explained_variance'][:]
            else:
                print('Precomputing ZCA...')
                self.precomputing = True
                batch_size = self.config['batch_size']
                self.config['batch_size'] = 128
                self.zca.n_samples_train = self.config['zca_epochs'] * len(self.videos) * self.config['frames']
                for epoch in range(self.config['zca_epochs']):
                    for b in tqdm.tqdm(range(len(self))):
                        self[b]
                self.precomputing = False
                self.config['batch_size'] = batch_size
                with h5py.File(filename, 'w') as f:
                    f.create_dataset('components', data=self.zca.components)
                    f.create_dataset('singular_values', data=self.zca.singular_values)
                    f.create_dataset('mean', data=self.zca.mean)
                    f.create_dataset('var', data=self.zca.var)
                    f.create_dataset('explained_variance', data=self.zca.explained_variance)

    def get_frames(self, video):
        try:
            frames = [f for f in os.listdir(self.config['path'] + '/' + video.split('/')[0]) if ('jpg' in f and video.split('/')[1][:-4] in f)]
        except:
            frames = [f for f in os.listdir(self.config['path'] + '/' + video) if ('jpg' in f)]
        frames = np.sort(frames).tolist()
        return frames        

    def build_frames(self, video, meta, frame_start=None, frame_end=None):
        """Given a video name, build our sequence."""
        frame_count, height, width = meta
        if frame_start == None:
            frame_start = 0
        elif frame_start >= frame_count:
            return np.array([], dtype=self.config['dtype'])
        if frame_end == None:
            frame_end = frames
        elif frame_end >= frame_count:
            frame_end = frame_count

        if 'size_crop' in self.config.keys():
            row_start = np.random.randint(height - self.config['size_crop'])
            col_start = np.random.randint(width - self.config['size_crop'])
            row_end = row_start + self.config['size_crop']
            col_end = col_start + self.config['size_crop']
        else:
            row_start, col_start, row_end, col_end = 0, 0, height, width
        frames = self.get_frames(video)
        imgs = []
        for j in range(frame_start, frame_end):
            try: 
                img = imread(self.config['path'] + '/' + video.split('/')[0] + '/' + frames[j])
            except:
                img = imread(self.config['path'] + '/' + video + '/' + frames[0])
            img = img[row_start:row_end, col_start:col_end]
            img = imresize(img, (self.config['size'], self.config['size']))
            if self.grayscale is True:
                img = np.dot(img[...,:3], [0.299, 0.587, 0.114])[:, :, None]
            imgs.append(img)
        imgs = np.array(imgs, dtype=self.config['dtype']) / 255.
        return imgs

