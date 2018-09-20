import preprocess as pp
import numpy as np

def modelInit(model):
    layerSize = [int(n) for n in model.split('-')]
    nHidden = len(layerSize) - 1

    weight = []
    for i in range(nHidden):
        weight.append(np.random.rand(layerSize[i+1], layerSize[i]))
    bias = []
    for i in range(nHidden):
        bias.append(np.random.rand(layerSize[i+1]))
    # ran = np.random.rand(3,2)
    # print(bias)

if __name__ == '__main__':
    data = pp.input('Flood_dataset.txt')
    trainSet, testSet = pp.kFolds(data,10)
    modelInit('8-6-2-1')
    # print(len(trainSet))
    # print('Hola')