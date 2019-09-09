from random import shuffle
import glob
import numpy as np
import h5py
import cv2
from math import ceil
import matplotlib.pyplot as plt


def build_h5_dataset(hdf5_path, data_order, train_x_l, val_x_l, test_x_l):
    # Selecciona el orden de los datos y elige las dimensiones apropiadas para almacenar las imágenes
    if data_order == 'th':
        train_shape = (train_x_l, 256, 256, 3)
        val_shape = (val_x_l, 256, 256, 3)
        test_shape = (test_x_l, 256, 256, 3)
    elif data_order == 'tf':
        train_shape = (train_x_l, 256, 256, 3)
        val_shape = (val_x_l, 256, 256, 3)
        test_shape = (test_x_l, 256, 256, 3)

        # Abrir un archivo HDF5 en modo escritura
    hdf5_file = h5py.File(hdf5_path, mode='w')
    # crear los datasets: train_img, val_img, test_img, train_mean
    hdf5_file.create_dataset("train_img", train_shape)
    hdf5_file.create_dataset("val_img", val_shape)
    hdf5_file.create_dataset("test_img", test_shape)

    # crear los datasets de etiquetas: train_labels, val_labels, test_labels
    hdf5_file.create_dataset("train_labels", (train_x_l,))
    hdf5_file.create_dataset("val_labels", (train_x_l,))
    hdf5_file.create_dataset("test_labels", (train_x_l,))

    return hdf5_file

healthy_scorch_path = "raw-color**/*.JPG"
shuffle_data = True

# leer las rutas de los archivos de la carpeta mix y asignar etiquetas
addrs = glob.glob(healthy_scorch_path)
labels = [0 if 'RS_HL' in addr else 1 for addr in addrs]  # 0 = Healthy, 1 = Leaf scorch

# para barajear las rutas de los archivos
if shuffle_data:
    c = list(zip(addrs, labels))
    shuffle(c)
    addrs, labels = zip(*c)

training_size = round(len(addrs) * 0.6)
test_size = round(len(addrs) * 0.2)
data_size = len(addrs)

# Divide las rutas de los archivos en: train_data (60%), val_data (20%), y test_data (20%)
train_addrs = addrs[:training_size]
train_labels = labels[:training_size]
val_addrs = addrs[training_size:training_size + test_size]
val_labels = labels[training_size:training_size + test_size]
test_addrs = addrs[training_size + test_size:]
test_labels = labels[training_size + test_size:]


# Actualiza la siguiente ruta acorde a su entorno de trabajo
hdf5_path = 'datasets/data_healthy.h5'  # Dirección donde queremos almacenar el archivo hdf5
data_order = 'tf'  # 'tf' para Tensorflow
hdf5_file=build_h5_dataset(hdf5_path, data_order, len(train_addrs), len(val_addrs), len(test_addrs))

print(f"Dimensiones train_img: {hdf5_file['train_img'].shape}")
