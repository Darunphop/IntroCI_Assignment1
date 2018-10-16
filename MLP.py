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
    #     print(y[i].shape)
    #     print(dW[i].shape)
    #     print(nB[i])
    # print(y)
    # print(weight)
    # print(len(nW))
    localGradient = [[] for i in range(len(y))]
    # print(localGradient)
    for i in reversed(range(1,len(activation))):
        for j in range(len(y[i-1])):
            # print(y[i-2][j])
            # print(nW)
            for k in range(len(y[i-1][j])):
                o = y[i-1][j][k]
                inv = act.activate(o, activation[i], True)
                if i == len(activation)-1:
                    error = d[j][k] - o
                    localGradient[i-1].append(error*inv)
                else:
                    # print(y[i][j])
                    sum = 0.0
                    for l in range(len(y[i][j])):
                        # print(weight[i][l][k])
                        sum += localGradient[i][l]*weight[i][l][k]
                    localGradient[i-1].append(sum*inv)
                # print(i,j,k)
                # print(nW[i-1][k])
                # print(localGradient[i-1][k])

                for l, lv in enumerate(nW[i-1][k]):
                    # print(y[i-1][j][k], 'prev o')
                    # print(i,j,k,l)
                    changeW = (momentum*dW[i-1][k][l]) + (learnRate*localGradient[i-1][k]*y[i-1][j][k])
                    nW[i-1][k][l] = weight[i-1][k][l] + changeW
                    dW[i-1][k][l] = changeW

                changeB = (momentum*dW[i-1][k][l]) + (learnRate*localGradient[i-1][k])
                nB[i-1][k] = bias[i-1][k] + changeB
                dB[i-1][k] = changeB


        # print(i)
        pass
    # print(localGradient)
    # print(weight)
    # print(nW)
    # print(dW)
    # print(nB)
    return nW, nB, dW, dB

def mse(y, d):
    res = 0.0
    dCp = np.asarray(d)
    yCp = y[-1:][0]
    for i in range(len(yCp)):
        sum = 0.0
        for j in range(len(yCp[i])):
            sum += (yCp[i][j] - dCp[i][j])**2
        res += sum / len(yCp)
    return res / len(yCp)

if __name__ == '__main__':
    pass