import numpy as np

def input(input):
    res = []
    with open(input, 'r') as inputFile:
        res = [list(map(int, line.rstrip().split('\t'))) for line in inputFile]
    return res

def normalize(x, denorm=False):
    #max 628 min 95 || 0 - 700
    MAX = 700
    MIN = 0
    if denorm:
        return (x * (MAX - MIN) + MIN)
    return (x - MIN) / (MAX - MIN)

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

if __name__ == 'preprocess':
    normalRange = np.vectorize(normalize)

if __name__ == '__main__':
    data = [[1,2,3,4],[4,5,6,7]]