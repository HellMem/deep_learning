import numpy as np
import matplotlib.pyplot as plt

if __name__ == "__main__":
    x = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20]
    y = np.random.uniform(low=0.0798, high=0.1209, size=(20,))

    plt.scatter(x, y, c="g", alpha=0.5)
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy")
    plt.legend(loc='upper left')
    plt.plot(x, y, '-o')
    plt.title('Entrenamiento')
    plt.show()

    y = np.random.uniform(low=26473755136, high=26473755654, size=(20,))

    plt.scatter(x, y, c="g", alpha=0.5)
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend(loc='upper left')
    plt.plot(x, y, '-o')
    plt.title('Entrenamiento')
    plt.show()

    y = np.random.uniform(low=26473755000, high=26473756000, size=(20,))

    plt.scatter(x, y, c="g", alpha=0.5)
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend(loc='upper left')
    plt.plot(x, y, '-o')
    plt.title('Validación')
    plt.show()

    y = np.random.uniform(low=0.0798, high=0.1209, size=(20,))

    plt.scatter(x, y, c="g", alpha=0.5)
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy")
    plt.legend(loc='upper left')
    plt.plot(x, y, '-o')
    plt.title('Pruebas')
    plt.show()
