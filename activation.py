import numpy as np

def sigmoid(x):
    return 1.0 / (1 + np.exp(0.0-x))

sigmoid = np.vectorize(sigmoid)

if __name__ == '__main__':
    sm = np.vectorize(sigmoid)
    print(sm([1,2,3]))