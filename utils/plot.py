import numpy as np
import matplotlib as mpl
mpl.use('Agg') 
import matplotlib.pyplot as plt


def plot(filepath, title, data, cmap=None, dpi=160):
    """Wrapper around pl.imshow"""
    if cmap is None:
        if data.shape[-1] == 3:
            cmap = plt.get_cmap('jet')
        elif data.shape[-1] == 1:
            cmap = plt.get_cmap('gray')
        elif data.shape[-1] == 2:
            data = data[..., 0]
            cmap = plt.get_cmap('gray')
        else:
            raise NotImplementedError('Number of image channels must be 1 or 3!')
    if data.shape[-1] == 1:
        data = np.reshape(data, data.shape[:-1])
    plt.figure(figsize=(30, 90), dpi=dpi)
    plt.imshow(data, vmin=0, vmax=1, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.axis('off')
    plt.savefig(filepath, bbox_inches='tight', dpi=dpi)
    plt.close() 


def make_mosaic(imgs, nrows, ncols, border=1, clip=False):
    """
    Given a set of images with all the same shape, makes a
    mosaic with nrows and ncols
    """
    nimgs = imgs.shape[0]
    imshape = imgs.shape[1:]
    mosaic = np.zeros((nrows * imshape[0] + (nrows - 1) * border,
                       ncols * imshape[1] + (ncols - 1) * border,
                       imshape[2]),
                      dtype=np.float32)
    paddedh = imshape[0] + border
    paddedw = imshape[1] + border
    for i in range(nimgs):
        row = i % nrows
        col = int(np.floor(i / nrows))
        img = imgs[i]
        if clip == False:
            img = (img - np.min(img, keepdims=True)) / (np.max(img, keepdims=True) - np.min(img, keepdims=True) + 1e-7)
        else:
            img = np.clip(img, 0, 1)
        mosaic[row * paddedh:row * paddedh + imshape[0],
               col * paddedw:col * paddedw + imshape[1], :] = img
    return mosaic

