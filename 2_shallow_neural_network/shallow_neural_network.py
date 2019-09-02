import numpy as np
import h5py
import cv2
import matplotlib.pyplot as plt
from PIL import Image
from scipy import ndimage


def load_dataset():
    hdf5_train_file = h5py.File("datasets/train_data.h5")
    hdf5_test_file = h5py.File("datasets/test_data.h5")

    train_set_x = np.array(hdf5_train_file["train_set_x"][:])
    train_set_y = np.array(hdf5_train_file["train_set_y"][:])
    test_set_x = np.array(hdf5_test_file["test_set_x"][:])
    test_set_y = np.array(hdf5_test_file["test_set_y"][:])

    list_classes = np.array(hdf5_test_file["list_classes"][:])

    hdf5_train_file.close()
    hdf5_test_file.close()

    return train_set_x, train_set_y, test_set_x, test_set_y, list_classes


def sigmoid(z):
    return (1 / (1 + np.exp(-z)))


def init_parameters(dim):
    w = np.zeros([dim, 1])
    return [w, 0]


def fb_propagation(weights, bias, X, Y):
    m = X.shape[1]

    Z = np.dot(weights.T, X) + bias

    A = sigmoid(Z)

    cost = 1. / m * (np.dot(Y, np.log(A).T) + np.dot((1 - Y), np.log(1 - A).T))

    dw = 1. / m * (np.dot(X, (A - Y).T))
    db = 1. / m * (np.sum(A - Y))

    gradients = {"dw": dw, "db": db}

    return gradients, cost


def optimization(X, Y, w, b, iterations, learning_rate):
    costs = []

    for i in range(iterations):
        gradients, cost = fb_propagation(w, b, X, Y)

        dw = gradients["dw"]
        db = gradients["db"]

        w = w - learning_rate * dw
        b = b - learning_rate * db

        if i % 100 == 0:
            costs.append(cost)

        parameters = {
            "w": w,
            "b": b
        }

        gradients = {
            "dw": dw,
            "db": db
        }

    return parameters, gradients, costs


if __name__ == "__main__":
    [train_set_x, train_set_y, test_set_x, test_set_y, list_classes] = load_dataset()

    print(train_set_x.shape)
    print(train_set_y.shape)
    print(test_set_x.shape)
    print(test_set_y.shape)
    print(list_classes.shape)

    train_m = train_set_x.shape[0]
    test_m = test_set_x.shape[0]
    img_dim = train_set_x.shape[1]

    print("Training size: ", train_m)
    print("Test size: ", test_m)
    print("Image Dimension: ", img_dim)

    train_set_x = train_set_x.reshape(img_dim * img_dim * 3, -1).T
    train_set_x = train_set_x / 255.

    test_set_x = test_set_x.reshape(img_dim * img_dim * 3, -1).T
    test_set_x = test_set_x / 255.

    dim = img_dim * img_dim * 3
    w, b = init_parameters(dim)

    X = np.array([[1, 2], [3, 4], [5, 6]])
    Y = np.array([[1, 0]])
    w = np.array([[1], [2], [3]])
    b = 1
    grads, cost = fb_propagation(X, Y, w, b)
    print("dw = " + str(grads["dw"]))
    print("db = " + str(grads["db"]))
    print("cost = " + str(cost))
