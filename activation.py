import numpy as np

def sigmoidO(x):
    return 1.0 / (1 + np.exp(0.0-x))

sigmoid = np.vectorize(sigmoidO)

def activate(x, func):
    res = x
    if func == 's':
        # print('in', res)
        res = sigmoid(x)
        # print('out', res)

    return res

if __name__ == '__main__':
    sm = np.vectorize(sigmoid)
    print(sm([1,2,3]))