import numpy as np

def input(input):
    res = []
    with open(input, 'r') as inputFile:
        res = [line.rstrip().split('\t') for line in inputFile]
    return res

def kFolds(data, k=1):
    trainSet = [[] for i in range(k)]
    testSet = [[] for i in range(k)]
    dataSize = len(data)
    binSize = int(dataSize / k)
    remainSize = dataSize % k

    np.random.shuffle(data)

    for i in range(k):
        trainSet[i].extend(data[0:(i)*binSize])
        trainSet[i].extend(data[(i+1)*binSize:dataSize-remainSize])
        testSet[i].extend(data[i*binSize:(i+1)*binSize])
        if remainSize != 0:
            trainSet[i].extend(data[-(dataSize % k):])
    
    return trainSet, testSet
