import preprocess as pp
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
    print(bias)
    # print(weigth)
    for i in range(len(activation)-1):
        # print('tmpIN',tmp)
        tmp = np.dot(len(tmp)==0 and input or tmp, np.transpose(weigth[i]))
        # print('tmpOUT',tmp)
        # print("BEFORE")
        # print(tmp)
        for j in range(tmp.shape[0]):
            for k in range(tmp.shape[1]):
                tmp[j][k] += bias[i][k]
        for j in range(tmp.shape[0]):
            tmp[j] = act.activate(np.copy(tmp[j]), activation[i+1])
            pass
    # print(input[0])
    
    return res

if __name__ == '__main__':
    data = pp.input('Flood_dataset.txt')
    trainSet, testSet = pp.kFolds(data,10)
    w,b,a = modelInit('9x-3s-2s-5s')
    inpu = trainSet[0][:2]
    o = feedForward(inpu, w, b, a)
    # print(len(trainSet[0]))
    # print(trainSet[0][:1])
    # print(act.sigmoid([1,2,3]))
    # print(a)