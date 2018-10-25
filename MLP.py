import activation as act
import numpy as np

def modelInit(model):
    layerSize = [int(n[:-1]) for n in model.split('-')]
    activationLayer = [n[-1:] for n in model.split('-')]
    nHidden = len(layerSize) - 1

    weight = []
    for i in range(nHidden):
        weight.append(np.random.randn(layerSize[i+1], layerSize[i]))
        # weight.append(np.ones((layerSize[i+1], layerSize[i])) / 5**-2)
    bias = []
    for i in range(nHidden):
        bias.append(np.random.rand(layerSize[i+1]))

    return weight, bias, activationLayer

def feedForward(input, weigth, bias, activation):
    res = []
    tmp = []

    for i in range(len(activation)-1):
        tmp = np.dot(len(tmp)==0 and input or tmp, np.transpose(weigth[i]))
        for j in range(tmp.shape[0]):
            for k in range(tmp.shape[1]):
                tmp[j][k] += bias[i][k]
        tmp = act.activate(np.copy(tmp), activation[i+1])
        res.append(np.asarray(tmp))
    return res

def backpropagate(input,y, weight, dWo, dBo, bias, activation, d, learnRate, momentum):
    # res = []
    # nW = [np.zeros(weight[i].shape) for i in range(len(weight))]
    nW = weight.copy()
    # nB = [np.zeros(bias[i].shape) for i in range(len(bias))]
    nB = bias.copy()
    if len(dWo) == 0:
        dWo = [np.zeros(weight[i].shape) for i in range(len(weight))]
    if len(dBo) == 0:
        dBo = [np.zeros(bias[i].shape) for i in range(len(bias))]
    dW = [np.zeros(weight[i].shape) for i in range(len(weight))]
    dB = [np.zeros(bias[i].shape) for i in range(len(bias))]


    localGradient = [[] for i in range(len(y))]
    # print(nW)
    # print(dW)
    # print('y',y[-1])
    # print('d',d)
    # error = d - y[-1]
    # print('error', error)
    # print(act.activate(y[-1], activation[-1], True))
    # localGradient[-1] = (act.activate(y[-1], activation[-1], True) * error).T
    # print('locG',localGradient)

    for i in reversed(range(len(y))):
        # print(i)
        if i+1 == len(y):
            error = d - y[i]
            localGradient[i] = (act.activate(y[i], activation[i], True) * error).T
        else:
            localGradient[i] = (act.activate(y[i], activation[-1], True) * (np.dot(localGradient[i+1].T, weight[i+1]))).T
        # print(localGradient)
        # if i == 0:
        #     dW[i] = learnRate * np.dot(localGradient[i], input)
        # else:
        #     dW[i] = learnRate * np.dot(localGradient[i], y[i-1])
        dW[i] = (momentum*dWo[i]) + (learnRate * np.dot(localGradient[i], i==0 and input or y[i-1]) / len(input))
        dB[i] = (momentum*dBo[i]) + np.average(learnRate * localGradient[i], axis=1)
        nW[i] += dW[i]
        nB[i] += dB[i]
        # print('dW',dW)
        # print('dB',dB)
    # print('locG',localGradient)

    # print('loG c',localGradient[1:][1].shape)
    # print('y[1:', y[:-1][1].shape)
    # dw = learnRate * ( np.dot(localGradient[1:], y[:-1]) )

    return nW, nB, dW, dB

def mse(y, d):
    res = 0.0
    # print('d',d)
    dCp = np.asarray(d)
    # print('dCp',dCp)

    yCp = y
    # print('yCp',yCp)
    # print('y', y)
    # for i in range(len(yCp)):
    #     sum = 0.0
    #     for j in range(len(yCp[i])):
    #         sum += (yCp[i][j] - dCp[i][j])**2
    #     res += sum / len(yCp)
    return np.average(np.abs(dCp - yCp))

if __name__ == '__main__':
    pass