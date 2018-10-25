import preprocess as pp
import MLP as mlp
import numpy as np
import sys
import copy

def training(trainingSet, testSet, epoch, w, b, a, learnRate, momentum, resAttr=1):
    inpu = np.delete(trainingSet, -resAttr, 1)
    inpu = pp.normalRange(inpu).tolist()
    d = np.delete(trainingSet, range(len(trainingSet[0])-resAttr), 1)
    d = pp.normalRange(d)

    for i in range(epoch):
            if i == 0:
                dW = []
                dB = []
            o = mlp.feedForward(inpu, w, b, a)
            if (i+1) % 100 == 0:
                print(i+1, testing(w,b,a,testSet))
            w,b,dW,dB = mlp.backpropagate(inpu,o,w,dW,dB,b,a,d,learnRate,momentum)

    return 0

def testing(w, b, a, data, resAttr=1):
    inpu = np.delete(data, -resAttr, 1)
    inpu = pp.normalRange(inpu).tolist()
    d = np.delete(data, range(len(data[0])-resAttr), 1)


    o = mlp.feedForward(inpu, w, b, a)[-1]
    o = np.round(pp.normalize(o, denorm=True))

    return mlp.mse(o,d)
    
if __name__ == '__main__':
    if sys.argv[1] == 'exp1':
        data = pp.input('Flood_dataset.txt')
        trainSet, testSet = pp.kFolds(data,10)
        w,b,a = mlp.modelInit('8x-5s-1s')
        chunk = trainSet[0]
        inpu = np.delete(chunk, -1, 1)
        inpu = pp.normalRange(inpu).tolist()
        d = np.delete(chunk, range(len(chunk[0])-1), 1)
        d = pp.normalRange(d)

        for i in range(10000):
            if i == 0:
                dW = []
                dB = []
            o = mlp.feedForward(inpu, w, b, a)
            if (i+1) % 100 == 0:
                print(i+1, testing(w,b,a,testSet[0]))
            w,b,dW,dB = mlp.backpropagate(inpu,o,w,dW,dB,b,a,d,0.001,0.5)
    
    elif sys.argv[1] == 'exp2':
        pass
    else:
        pass
        
    
