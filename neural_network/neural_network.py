import numpy as np
import copy

def load_dataset():
    np.random.seed(1)
    m = 400  # número de ejemplos
    N = int(m / 2)  # número de puntos por clase
    D = 2  # dimensionalidad
    X = np.zeros((m, D))  # matriz de datos (cada renglón es un ejemplo)
    Y = np.zeros((m, 1), dtype='uint8')  # vector de etiquetas ( 1 azul, 0 - rojo)
    a = 4  # max ray of the flower
    for j in range(2):
        ix = range(N * j, N * (j + 1))
        t = np.linspace(j * 3.12, (j + 1) * 3.12, N) + np.random.randn(N) * 0.2  # theta
        r = a * np.sin(4 * t) + np.random.randn(N) * 0.2  # radius
        X[ix] = np.c_[r * np.sin(t), r * np.cos(t)]
        Y[ix] = j
    X = X.T
    Y = Y.T
    return X, Y


def layer_sizes_test():
    np.random.seed(1)
    X_test = np.random.randn(2, 3)
    Y_test = np.random.randn(1, 3)
    return X_test, Y_test


def layer_sizes(X, Y):
    n_x = X.shape[0]
    n_y = Y.shape[0]
    return n_x, n_y


def init_parameters(n_x, n_h, n_y):
    np.random.seed(2)
    W1 = np.random.randn(n_x, n_h) * 0.01
    b1 = np.zeros((n_h, 1))

    W2 = np.random.randn(n_h, n_y) * 0.01
    b2 = np.zeros((n_y, 1))

    parameters = {"W1": W1, "b1": b1, "W2": W2, "b2": b2}

    return parameters


def sigmoid(x):
    s = 1 / (1 + np.exp(-x))
    return s


def forward_prop(X, weights, biases):
    H = X
    for i in range(len(weights)):
        weight = weights[i]
        bias = biases[i]
        Zi = np.dot(weight.T, H) + bias
        H = sigmoid(Zi)

    return H


def back_prop(X, Y, weights, biases):
    error_weights = [np.zeros(weight.shape) for weight in weights]
    error_biases = [np.zeros(bias.shape) for bias in biases]

    for x, y in zip(X.T, Y.T):
        delta_errors_weights = [np.zeros(weight.shape) for weight in weights]
        delta_errors_biases = [np.zeros(bias.shape) for bias in biases]

        activation = x
        activations = [x]

        zs = []

        z1 = np.dot(activation, weights[0]) + biases[0]
        zs.append(z1)
        activation = sigmoid(z1)
        activations.append(activation)

        z2 = np.dot(activation, weights[1]) + biases[1]
        zs.append(z2)
        activation = np.tanh(z2)
        activations.append(activation)

        delta = (activations[1] - y)
        delta_errors_biases[1] = delta
        delta_errors_weights[1] = np.dot(activations[1].transpose(), delta)

        error_weights = [ew + dew for ew, dew in zip(error_weights, delta_errors_weights)]
        error_biases = [eb + deb for eb, deb in zip(error_biases, delta_errors_biases)]

    return error_weights, error_biases


def cost_function(X, Y, weights, biases):
    cost = 0
    epsilon = 1e-5

    for x, y in zip(X.T, Y.T):
        predictions = forward_prop(x, weights, biases)

        part1_cost = y * np.log(predictions + epsilon)

        part2_cost = (1 - y) * np.log(1 - predictions + epsilon)

        cost += (part1_cost.sum() + part2_cost.sum())

    return -cost / len(X)


# Se ejecuta el gradiente descendiente para determinar el módelo óptimo
def train(X, y, weights, biases, alpha, iters):
    training_set_size = len(X)
    history = {'cost': [], 'iter': []}

    for i in range(iters):
        gradW, gradB = back_prop(X, y, weights, biases)
        weights_copy = copy.deepcopy(weights)
        weights = []
        for w, g in zip(weights_copy, gradW):
            w = np.add(w, - (g * alpha / training_set_size))
            weights.append(w)

        biases_copy = copy.deepcopy(biases)
        biases = []
        for b, g in zip(biases_copy, gradB):
            b = np.add(b, - (g * alpha / training_set_size))
            biases.append(b)

        cost = cost_function(X, y, weights, biases)
        history['cost'].append(cost)
        history['iter'].append(i)

        if i % 10 == 0:
            print("iter: " + str(i) + " cost: " + str(cost))

    return [weights, biases, history]

def net_model_241():
    X, Y = load_dataset()
    X_test, Y_test = layer_sizes_test()
    n_x, n_y = layer_sizes(X_test, Y_test)
    learning_rate = 1.2
    iterations = 1000

    parameters = init_parameters(n_x, 4, n_y)
    weights = [parameters["W1"], parameters["W2"]]
    biases = [parameters["b1"], parameters["b2"]]
    print("Para n_h = 4")
    train(X, Y, weights, biases, learning_rate, iterations)

    parameters = init_parameters(n_x, 16, n_y)
    weights = [parameters["W1"], parameters["W2"]]
    biases = [parameters["b1"], parameters["b2"]]
    print("Para n_h = 16")
    train(X, Y, weights, biases, learning_rate, iterations)

    parameters = init_parameters(n_x, 32, n_y)
    weights = [parameters["W1"], parameters["W2"]]
    biases = [parameters["b1"], parameters["b2"]]
    print("Para n_h = 32")
    train(X, Y, weights, biases, learning_rate, iterations)

    parameters = init_parameters(n_x, 64, n_y)
    weights = [parameters["W1"], parameters["W2"]]
    biases = [parameters["b1"], parameters["b2"]]
    print("Para n_h = 64")
    train(X, Y, weights, biases, learning_rate, iterations)

    parameters = init_parameters(n_x, 128, n_y)
    weights = [parameters["W1"], parameters["W2"]]
    biases = [parameters["b1"], parameters["b2"]]
    print("Para n_h = 128")
    train(X, Y, weights, biases, learning_rate, iterations)

    parameters = init_parameters(n_x, 256, n_y)
    weights = [parameters["W1"], parameters["W2"]]
    biases = [parameters["b1"], parameters["b2"]]
    print("Para n_h = 256")
    train(X, Y, weights, biases, learning_rate, iterations)


if __name__ == "__main__":
    net_model_241()


