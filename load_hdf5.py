import h5py
from os.path import join
import numpy as np
import scipy.misc
import os

f1 = h5py.File('LLD-icon-sharp.hdf5','r+')

nb_picture = f1['data'].shape[0]

for i in range(nb_picture):
    folder_path = join('cluster', str(f1['labels']['resnet']['rc_32'][i]))
    os.makedirs(folder_path, exist_ok=True)
    file_path =join(folder_path, str(i)+'.png')
    scipy.misc.imsave(file_path, np.moveaxis(f1['data'][i], 0, -1))