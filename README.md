# Group k-Sparse Temporal Convolutional Neural Networks: Pre-training for Video Classification
## Source code for reproducing the experiments described in the paper

### Running the code
Reproduction should be as easy as executing this in the root folder (after installing the dependencies with pip3):
```bash
ipython3 experiments/mnistrotated.py with groupwtacrnn nospatial seed=123
```
In general:
```bash
ipython3 experiments/database.py with method optional_config seed=number
```
where method is either
* randominitcnn (Glorot Initialized Time Distributed CNN)
* randominitcrnn (Glorot Initialized Recurrent CNN)
* denoisingcnn (Denoising Time Distributed CNN Autoencoder)
* denoisingcrnn (Denoising Recurrent CNN Autoencoder)
* wtacnn (Winner-Take-All Time Distributed CNN Autoencoder)
* wtacrnn (Winner-Take-All Recurrent CNN Autoencoder)
* vgg19 (ImageNet Pretrained Time Distributed VGG19)
* groupwtacnn (Group k-Sparse Time Distributed CNN Autoencoder)
* groupwtacrnn (Group k-Sparse Recurrent CNN Autoencoder)

and optional_config is either nothing (both spatial and lifetime sparsity are enabled) or
* nospatial (disable spatial sparsity)
* nolifetime (disable lifetime sparsity)

We used seed=123 in all experiments.

### Dependencies
* numpy
* scipy
* sklearn
* keras
* tensorflow
* sacred
* matplotlib
* pil
* h5py
* tqdm
