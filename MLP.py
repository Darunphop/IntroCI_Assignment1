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

def backpropagate(y, weight, dW, dB, bias, activation, d, learnRate, momentum):
    res = []
    nW = [np.zeros(weight[i].shape) for i in range(len(weight))]
    nB = [np.zeros(bias[i].shape) for i in range(len(bias))]
    if len(dW) == 0:
        dW = [np.zeros(weight[i].shape) for i in range(len(weight))]
    if len(dB) == 0:
        dB = [np.zeros(bias[i].shape) for i in range(len(bias))]

    
    # for i in range(len(weight)):
    #     print(weight[i].shape)
    #     # print(dW[i].shape)
    #     print(nB[i])
    # print(y)
    print(len(nW))
    localGradient = [[] for i in range(len(y))]
    # print(localGradient)
    for i in reversed(range(1,len(activation))):
        if i == len(activation)-1:
            for j in range(len(y[i-1])):
                for k in range(len(y[i-1][j])):
                    o = y[i-1][j][k]
                    error = d[j][k] - o
                    inv = act.activate(o, activation[i], True)
                    localGradient[i-1].append(error*inv)

                    # print(nW[i-1][k])
                    # print(localGradient[i-1][k])
                    for l, lv in enumerate(nW[i-1][k]):
                        # print(y[i-2][j][l], 'prev o')
                        changeW = (momentum*dW[i-1][k][l]) + (learnRate*localGradient[i-1][k]*y[i-2][j][l])
                        nW[i-1][k][l] = weight[i-1][k][l] + changeW

                    changeB = (momentum*dW[i-1][k][l]) + (learnRate*localGradient[i-1][k])
                    nB[i-1][k] = bias[i-1][k] + changeB
        else:
            pass


        # print(i)
        pass
    # print(localGradient)
    # print(weight)
    # print(nW)
    print(nB)
    return res

if __name__ == '__main__':
    pass