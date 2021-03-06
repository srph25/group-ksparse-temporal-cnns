# Group k-Sparse Temporal Convolutional Neural Networks: Pre-training for Video Classification
## Python source code for reproducing the experiments described in the paper
[Paper (IEEE Xplore)](https://ieeexplore.ieee.org/abstract/document/8852057)\
\
Code is mostly self-explanatory via file, variable and function names; but more complex lines are commented.\
Designed to require minimal setup overhead, using as much Keras and sacred integration and reusability as possible.

### Installing dependencies
**Installing Python 3.7.9 on Ubuntu 20.04.2 LTS:**
```bash
sudo add-apt-repository ppa:deadsnakes/ppa
sudo apt-get update
sudo apt-get install python3.7
```
**Installing CUDA 10.0:**
```bash
wget https://developer.nvidia.com/compute/cuda/10.0/Prod/local_installers/cuda_10.0.130_410.48_linux
sudo bash cuda_10.0.130_410.48_linux --override
echo 'export PATH=/usr/local/cuda-10.0/bin:$PATH' >> ~/.bashrc
echo 'export LD_LIBRARY_PATH=/usr/local/cuda-10.0/lib64:$LD_LIBRARY_PATH' >> ~/.bashrc
source ~/.bashrc
```
**Installing cuDNN 7.6.5:**
```bash
wget http://people.cs.uchicago.edu/~kauffman/nvidia/cudnn/cudnn-10.0-linux-x64-v7.6.5.32.tgz
# if link is broken, login and download from nvidia:
# https://developer.nvidia.com/compute/machine-learning/cudnn/secure/7.6.5.32/Production/10.0_20191031/cudnn-10.0-linux-x64-v7.6.5.32.tgz
tar -xvzf cudnn-10.0-linux-x64-v7.6.5.32.tgz
sudo cp -r cuda/include/* /usr/local/cuda-10.0/include/
sudo cp -r cuda/lib64/* /usr/local/cuda-10.0/lib64/
```
**Installing Python packages with pip:**
```bash
python3.7 -m pip install h5py==2.10.0 ipython==7.16.1 keras==2.2.4 matplotlib==3.3.2 numpy==1.19.2 pillow==8.1.0 pywavelets==1.1.1 sacred==0.8.2 scikit-learn==0.23.2 scipy==1.5.2 tensorflow-gpu==1.14.0 tqdm==4.56.0
```

### Running the code
Reproduction should be as easy as executing this in the root folder (after installing all dependencies):
```bash
python3.7 -m IPython experiments/mnistrotated.py with groupwtacrnn nospatial seed=123
```
In general:
```bash
python3.7 -m IPython experiments/dataset.py with algorithm optional_config seed=number
```

where `dataset` is either:
* `mnistrotated` : the Rotated MNIST video set, artificially generated by rotating and picking the top left corner,
* `cifar10scanned` : the Scanned CIFAR-10 video set, artificially generated by sliding a window,
* `coil100` : the COIL-100 natural video set, placing objects on turning table;
* `necanimal` : the NEC Animal natural video set, placing animal figures on turning table;

`algorithm` is either:
* `wtacnn` : Winner-Take-All (WTA) Time Distributed CNN Autoencoder,
* `wtacrnn` : Winner-Take-All (WTA) Recurrent CNN Autoencoder,
* `randominitcnn` : Glorot Initialized Time Distributed CNN,
* `randominitcrnn` : Glorot Initialized Recurrent CNN,
* `denoisingcnn` : Denoising Time Distributed CNN Autoencoder,
* `denoisingcrnn` : Denoising Recurrent CNN Autoencoder,
* `vgg19` : ImageNet Pretrained Time Distributed VGG19,
* `groupwtacnn` : Group k-Sparse Time Distributed CNN Autoencoder,
* `groupwtacrnn` : Group k-Sparse Recurrent CNN Autoencoder;

and `optional_config` is either nothing (both spatial and lifetime sparsity enabled by default), or:
* `nospatial` : disable spatial sparsity,
* `nolifetime` : disable lifetime sparsity.

`seed` : `123` in all of our experiments, should yield very similar numbers as in the table of our paper


### Directory and file structure:
algorithms/\
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;keraswtacnn.py : base class, the original WTA autoencoder baseline method\
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;keraswtacrnn.py : subclass, WTA with recurrent connections\
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;kerasrandominitcnn.py : subclass, no pretraining baseline method\
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;kerasrandominitcrnn.py : subclass, no pretraining with recurrent connections\
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;kerasdenoisingcnn.py : subclass, input dropout autoencoder baseline method\
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;kerasdenoisingcrnn.py : subclass, input dropout with recurrent connections\
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;kerasvgg19.py : subclass, imagenet pretraining baseline method\
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;kerasgroupwtacnn.py : subclass, our group k-sparse autoencoder\
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;kerasgroupwtacrnn.py : subclass, our group k-sparse autoencoder with recurrent connections\
datasets/\
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;mnistrotated.py : base class, loads Rotated MNIST data set and generates given number of labeled samples\
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;cifar10scanned.py : subclass, same but for Scanned CIFAR-10\
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;coil100.py : subclass, same but for COIL-100\
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;necanimal.py : subclass, same but for NEC Animal\
experiments/\
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;mnistrotated.py : config file for hyperparameters, loads Rotated MNIST data set and an algorithm,\
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;conducts experiment\
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;cifar10scanned.py : same, but for Scanned CIFAR-10\
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;coil100.py : same, but for COIL-100\
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;necanimal.py : same, but for NEC Animal\
results/ : experimental results will be saved to this directory with sacred package\
utils/\
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;layers.py : custom Keras layer classes, including\
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;`ConvMinimalRNN2D` : the convolutional minimal recurrent layer\
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;ops.py : custom Keras/Tensorflow operations, including\
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;`n_p` : p-norm computation\
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;`group_norms` : grouped p-norm computation\
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;`ksparse` : top-k masking activation function\
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;`group_ksparse` : our grouped top-k masking activation function\
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;pil.py : functions for backwards compatibility for saving all kinds of figures\
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;plot.py : functions for saving video frame figures\
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;preprocessing.py : functions for ZCA whitening\
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;utils.py : additional things, including\
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;`VideoSequence` : Keras Sequence subclass generating random videos


### Citation:
```latex
@inproceedings{milacski2019group,
  title={Group k-sparse temporal convolutional neural networks: unsupervised pretraining for video classification},
  author={Milacski, Zolt{\'a}n {\'A} and P{\'o}czos, Barnab{\'a}s and L{\H{o}}rincz, Andr{\'a}s},
  booktitle={2019 International Joint Conference on Neural Networks (IJCNN)},
  pages={1--10},
  year={2019},
  organization={IEEE}
}
```


### Contact:
In case of any questions, feel free to create an issue here on GitHub, or mail me at [srph25@gmail.com](mailto:srph25@gmail.com).

