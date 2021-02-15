import numpy as np
import os
os.environ['TF_FORCE_GPU_ALLOW_GROWTH']='true'
import datetime
from sacred import Experiment
from sacred.observers import FileStorageObserver
from datasets.necanimal import NECAnimalDataset
from datasets.coil100 import COIL100Dataset
from algorithms.kerasdenoisingcnn import KerasDenoisingCNN
from algorithms.kerasdenoisingcrnn import KerasDenoisingCRNN
from algorithms.keraswtacnn import KerasWTACNN
from algorithms.keraswtacrnn import KerasWTACRNN
from algorithms.kerasgroupwtacnn import KerasGroupWTACNN
from algorithms.kerasslowgroupwtacnn import KerasSlowGroupWTACNN
from algorithms.kerasgroupwtacrnn import KerasGroupWTACRNN
from algorithms.kerasrandominitcnn import KerasRandomInitCNN
from algorithms.kerasrandominitcrnn import KerasRandomInitCRNN
from algorithms.kerasvgg19 import KerasVGG19


name = os.path.basename(__file__).split('.')[0]
ex = Experiment(name)
dt = datetime.datetime.now()
results_dir = 'results/' + name + '/' + '{y:04d}{mo:02d}{d:02d}{h:02d}{mi:02d}{s:02d}_{p:05d}'.format(y=dt.year, mo=dt.month, d=dt.day, h=dt.hour, mi=dt.minute, s=dt.second, p=os.getpid()) + '_' + os.uname()[1]
ex.observers.append(FileStorageObserver.create(results_dir))


@ex.config
def cfg():
    _data = {'num_labeled': [2, 4, 8, 16, 32],
             'path': 'datasets/animalclean',
             'size': 32,
             #'frames': 72,
             'zca_epsilon': 1e-3,
             'pad_value': -1,
             'imagenet_preprocessing': False,
             'dtype': 'float32'}
    _algo = {'batch_size': 10,#1,#30,
             'shape1': 79,#_data['frames'],
             'shape2': _data['size'],
             'shape3': _data['size'],
             'shape4': 3,
             'classes': 100,
             'filters': 128,
             'filter_size_enc': 5,
             'filter_size_dec': 7,
             'k_spatial': 1,
             'k_lifetime': 1,
             'pool_size': 2,
             'dropout': 0.3,
             'l2': 0.,
             'epochs': 300,#200,#50,
             'patience': 5,
             'lr': 1e-4,
             'momentum': 0.9,
             'clipnorm': 0.,
             'gpus': 1,
             'pad_value': _data['pad_value']}

@ex.named_config
def denoisingcnn():
    _algo = {'mode': 'denoisingcnn'}

@ex.named_config
def denoisingcrnn():
    _algo = {'mode': 'denoisingcrnn'}

@ex.named_config
def wtacnn():
    _algo = {'mode': 'wtacnn'}
    
@ex.named_config
def wtacrnn():
    _algo = {'mode': 'wtacrnn',
             'clipnorm': 1.}

@ex.named_config
def groupwtacnn():
    _algo = {'mode': 'groupwtacnn',
             'groups': 32}

@ex.named_config
def slowgroupwtacnn():
    _algo = {'mode': 'slowgroupwtacnn',
             'groups': 32,
             'l1_slow': 1e-4}

@ex.named_config
def groupwtacrnn():
    _algo = {'mode': 'groupwtacrnn',
             'groups': 32,
             'clipnorm': 1.}

@ex.named_config
def randominitcnn():
    _algo = {'mode': 'randominitcnn'}

@ex.named_config
def randominitcrnn():
    _algo = {'mode': 'randominitcrnn',
             'clipnorm': 1.}

@ex.named_config
def vgg19():
    _data = {'zca_epsilon': None,
            'imagenet_preprocessing': True}
    _algo = {'mode': 'vgg19',
             'shape2': 32,
             'shape3': 32}

@ex.named_config
def nolifetime(_algo):
    _algo = {'k_lifetime': 10 * 79}#30 * 9}

@ex.named_config
def nospatial(_algo):
    _algo = {'k_spatial': 32 * 32}#16 * 16}

@ex.automain
def run(_data, _algo, _rnd, _seed):
    data_necanimal = NECAnimalDataset(config=_data)
    data_coil100 = COIL100Dataset(config=dict(list(_data.items()) + [('path', 'datasets/coil-100')]))
    if _algo['mode'] == 'denoisingcnn':
        alg = KerasDenoisingCNN(results_dir=results_dir, config=_algo)
    elif _algo['mode'] == 'denoisingcrnn':
        alg = KerasDenoisingCRNN(results_dir=results_dir, config=_algo)
    elif _algo['mode'] == 'wtacnn':
        alg = KerasWTACNN(results_dir=results_dir, config=_algo)
    elif _algo['mode'] == 'wtacrnn':
        alg = KerasWTACRNN(results_dir=results_dir, config=_algo)
    elif _algo['mode'] == 'groupwtacnn':
        alg = KerasGroupWTACNN(results_dir=results_dir, config=_algo)
    elif _algo['mode'] == 'slowgroupwtacnn':
        alg = KerasSlowGroupWTACNN(results_dir=results_dir, config=_algo)
    elif _algo['mode'] == 'groupwtacrnn':
        alg = KerasGroupWTACRNN(results_dir=results_dir, config=_algo)
    elif _algo['mode'] == 'randominitcnn':
        alg = KerasRandomInitCNN(results_dir=results_dir, config=_algo)
    elif _algo['mode'] == 'randominitcrnn':
        alg = KerasRandomInitCRNN(results_dir=results_dir, config=_algo)
    elif _algo['mode'] == 'vgg19':
        alg = KerasVGG19(results_dir=results_dir, config=_algo)
    alg.build()

    result = []
    alg.train(data_necanimal.X)#_train)#, X_val=data.X_val)
    alg.plot_predictions(data_necanimal.X[:8],#_train[:8], 
                         filepath=results_dir + '/videos_necanimal.png', title='Training Input and Reconstructed Videos')
    alg.config['shape1'] = data_coil100.X.shape[1]
    res = []
    res.append(alg.test(data_necanimal.X))
    alg.build()
    alg.autoencoder_base.load_weights(results_dir + '/autoencoder.hdf5')
    alg.plot_predictions(data_coil100.X[:8],#_train[:8], 
                         filepath=results_dir + '/videos_coil100.png', title='Test Input and Reconstructed Videos')
    #alg.plot_predictions(data.X_val[:8], filepath=results_dir + '/videos_val.png', title='Validation Input and Reconstructed Videos')
    #alg.plot_predictions(data.X_test[:8], filepath=results_dir + '/videos_test.png', title='Test Input and Reconstructed Videos')
    res.append(alg.test(data_coil100.X))
    result.append(res)#_train), #alg.test(data.X_val), 
                   #alg.test(data.X_test)])
    
    for num_labeled in _data['num_labeled']:
        data_coil100.generate_labeled_data(num_labeled)
        alg.config['shape1'] = data_coil100.X_train_labeled.shape[1]
        alg.build()
        alg.autoencoder_base.load_weights(results_dir + '/autoencoder.hdf5')
        alg.train_classifier(data_coil100.X_train_labeled, data_coil100.Y_train_labeled, X_val=data_coil100.X_val_labeled, Y_val=data_coil100.Y_val_labeled, stacked=True)
        res = []
        alg.config['shape1'] = data_coil100.X_train_labeled.shape[1]
        alg.build()
        alg.stacked_classifier_base.load_weights(results_dir + '/supervised_True.hdf5')
        res.append(alg.test_classifier(data_coil100.X_train_labeled, data_coil100.Y_train_labeled, stacked=True))
        res.append(alg.test_classifier(data_coil100.X_val_labeled, data_coil100.Y_val_labeled, stacked=True))
        alg.config['shape1'] = data_coil100.X_test.shape[1]
        alg.build()
        alg.stacked_classifier_base.load_weights(results_dir + '/supervised_True.hdf5')
        res.append(alg.test_classifier(data_coil100.X_test, data_coil100.Y_test, stacked=True))
        result.append(res)
        alg.config['shape1'] = data_coil100.X_train_labeled.shape[1]
        alg.build()
        alg.endtoend_classifier_base.load_weights(results_dir + '/supervised_True.hdf5')
        alg.train_classifier(data_coil100.X_train_labeled, data_coil100.Y_train_labeled, X_val=data_coil100.X_val_labeled, Y_val=data_coil100.Y_val_labeled, stacked=False)
        res = []
        alg.config['shape1'] = data_coil100.X_train_labeled.shape[1]
        alg.build()
        alg.endtoend_classifier_base.load_weights(results_dir + '/supervised_False.hdf5')
        res.append(alg.test_classifier(data_coil100.X_train_labeled, data_coil100.Y_train_labeled, stacked=False))
        res.append(alg.test_classifier(data_coil100.X_val_labeled, data_coil100.Y_val_labeled, stacked=False))
        alg.config['shape1'] = data_coil100.X_test.shape[1]
        alg.build()
        alg.endtoend_classifier_base.load_weights(results_dir + '/supervised_False.hdf5')
        res.append(alg.test_classifier(data_coil100.X_test, data_coil100.Y_test, stacked=False))
        result.append(res)
    return result

