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
    err = [0.0 for i in range(len(y[-1][0]))]
    print(y[-1],'y')
    print(d,'d')
    print(y[-1] - np.asarray(d))
    err = np.sum(y[-1] - np.asarray(d), axis=0)
    print(err,'errr')

    print(activation)
    for i in range(len(activation)):
        # layer =
        pass
    return res

if __name__ == '__main__':
    pass