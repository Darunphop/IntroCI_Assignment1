import activation as act
import numpy as np

def modelInit(model):
    layerSize = [int(n[:-1]) for n in model.split('-')]
    activationLayer = [n[-1:] for n in model.split('-')]
    nHidden = len(layerSize) - 1

    weight = []
    for i in range(nHidden):
        weight.append(np.random.rand(layerSize[i+1], layerSize[i]))
    bias = []
    for i in range(nHidden):
        bias.append(np.random.rand(layerSize[i+1]))

    return weight, bias, activationLayer

def feedForward(input, weigth, bias, activation):
    res = []
    tmp = []

    print(activation)
    # print(bias)
    # print(weigth)
    for i in range(len(activation)-1):
        tmp = np.dot(len(tmp)==0 and input or tmp, np.transpose(weigth[i]))
        for j in range(tmp.shape[0]):
            for k in range(tmp.shape[1]):
                tmp[j][k] += bias[i][k]
        for j in range(tmp.shape[0]):
            tmp[j] = act.activate(np.copy(tmp[j]), activation[i+1])
        res.append(tmp)
    
    return res

def backpropagate(y, weight, bias, activation, d, learnRate, momentum):
    res = []
    err = [0.0 for i in range(len(d[0]))]
    print(err)
    for i in range(len(d)):
        if len(d[i]) > 1:
            for j in range(len(d[i])):
                y[-1:][i]
        else:
            err += (d[i] - y[-1][i])**2
    print(err)
    print(activation)
    for i in range(len(activation)):
        # layer =
        pass
    return res

if __name__ == '__main__':
    pass