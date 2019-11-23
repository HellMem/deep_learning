from random import shuffle
import glob
import numpy as np
import h5py
import cv2
from math import ceil
import matplotlib.pyplot as plt


def build_data_lists(paths, shuffle_data=True):
    animal_class = 0
    animal_addresses = []
    animal_classifications = []
    for path in paths:
        addrs = glob.glob(path)
        labels = [animal_class] * len(addrs)
        animal_addresses.extend(addrs)
        animal_classifications.extend(labels)
        animal_class += 1

    if shuffle_data:
        c = list(zip(animal_addresses, animal_classifications))
        shuffle(c)
        animal_addresses, animal_classifications = zip(*c)

    training_size = round(len(animal_addresses) * 0.8)
    test_size = round(len(animal_addresses) * 0.1)

    train_addrs = animal_addresses[:training_size - 1]
    train_labels = animal_classifications[:training_size - 1]
    val_addrs = animal_addresses[training_size:training_size + test_size]
    val_labels = animal_classifications[training_size:training_size + test_size]
    test_addrs = animal_addresses[training_size + test_size:]
    test_labels = animal_classifications[training_size + test_size:]

    return train_addrs, train_labels, val_addrs, val_labels, test_addrs, test_labels


def build_h5_dataset(hdf5_path, train_x_l, val_x_l, test_x_l):
    train_shape = (train_x_l, 256, 256, 3)
    val_shape = (val_x_l, 256, 256, 3)
    test_shape = (test_x_l, 256, 256, 3)

    # Abrir un archivo HDF5 en modo escritura
    hdf5_file = h5py.File(hdf5_path, mode='w')
    # crear los datasets: train_img, val_img, test_img, train_mean
    hdf5_file.create_dataset("train_img", train_shape, np.uint8)
    hdf5_file.create_dataset("val_img", val_shape, np.uint8)
    hdf5_file.create_dataset("test_img", test_shape, np.uint8)
    hdf5_file.create_dataset("train_mean", train_shape[1:])

    # crear los datasets de etiquetas: train_labels, val_labels, test_labels
    hdf5_file.create_dataset("train_labels", (train_x_l,), np.uint8)
    hdf5_file.create_dataset("val_labels", (val_x_l,), np.uint8)
    hdf5_file.create_dataset("test_labels", (test_x_l,), np.uint8)

    return hdf5_file


def load_images_to_h5_dataset(hdf5_file, train_addrs, val_addrs, test_addrs, train_labels, val_labels,
                              test_labels):
    hdf5_file["train_labels"][...] = train_labels
    hdf5_file["val_labels"][...] = val_labels
    hdf5_file["test_labels"][...] = test_labels

    train_shape = hdf5_file["train_img"].shape
    mean = np.zeros(train_shape[1:], np.float32)
    width = 256
    height = 256
    dim = (width, height)

    for i in range(len(train_addrs)):
        if i % 150 == 0 and i > 1:
            print(f"Datos de entrenamiento: {i}/{len(train_addrs)}")

        addr = train_addrs[i]
        img = cv2.imread(addr)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, dim, interpolation=cv2.INTER_AREA)

        hdf5_file["train_img"][i, ...] = img[None]
        mean += img / float(len(train_labels))

    for i in range(len(val_addrs)):
        if i % 150 == 0 and i > 1:
            print(f"Datos de validación: {i}/{len(val_addrs)}")

        addr = val_addrs[i]
        img = cv2.imread(addr)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, dim, interpolation=cv2.INTER_AREA)

        hdf5_file["val_img"][i, ...] = img[None]
        mean += img / float(len(val_labels))

    for i in range(len(test_addrs)):
        if i % 150 == 0 and i > 1:
            print(f"Datos de pruebas: {i}/{len(test_addrs)}")

        addr = test_addrs[i]
        img = cv2.imread(addr)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, dim, interpolation=cv2.INTER_AREA)

        hdf5_file["test_img"][i, ...] = img[None]
        mean += img / float(len(test_labels))

    # Guardemos la media
    hdf5_file["train_mean"][...] = mean
    return hdf5_file


def read_h5_dataset(hdf5_path, batch_size, batch_n):
    hdf5_file = h5py.File(hdf5_path, "r")

    data_num = hdf5_file["train_img"].shape[0]

    batches_list = list(range(int(ceil(float(data_num) / batch_size))))
    shuffle(batches_list)

    for n, i in enumerate(batches_list):
        i_s = i * batch_size  # Indice de la primer imagen en este lote
        i_e = min([(i + 1) * batch_size, data_num])  # Indice de la última imagen en este lote

        images = hdf5_file["train_img"][i_s:i_e]
        labels = hdf5_file["train_labels"][i_s:i_e]

        print(n + 1, '/', len(batches_list))
        print(f"Etiqueta: {labels[0]}")

        plt.imshow(images[0])
        plt.show()
        if n == (batch_n - 1):
            break

    return hdf5_file


if __name__ == "__main__":

    paths = ["raw-img/cane**/*.jpeg", "raw-img/cavallo**/*.jpeg", "raw-img/elefante**/*.jpeg",
             "raw-img/farfalla**/*.jpeg", "raw-img/gallina**/*.jpeg", "raw-img/gatto**/*.jpeg",
             "raw-img/mucca**/*.jpeg", "raw-img/pecora**/*.jpeg", "raw-img/ragno**/*.jpeg",
             "raw-img/scoiattolo**/*.jpeg"]


    #Para generar el archivo hdf5 

    train_addrs, train_labels, val_addrs, val_labels, test_addrs, test_labels = build_data_lists(paths)
    hdf5_path = 'datasets/animals.h5'
    hdf5_file = build_h5_dataset(hdf5_path, len(train_addrs), len(val_addrs), len(test_addrs))
    print(f"Dimensiones train_img: {hdf5_file['train_img'].shape}")
    load_images_to_h5_dataset(hdf5_file, train_addrs, val_addrs, test_addrs, train_labels, val_labels, test_labels)
    hdf5_file.close()

    #Para leer el archivo hdf5
    hdf5_path = 'datasets/animals.h5'
    batch_size = 50
    batch_n = 4
    hdf5_file = read_h5_dataset(hdf5_path, batch_size, batch_n)
    hdf5_file.close()
