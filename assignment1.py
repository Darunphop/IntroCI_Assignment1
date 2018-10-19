import preprocess as pp
import MLP as mlp
import numpy as np
import copy

def training(trainingSet, epoch, w, b, a, leanRate, momentum, resAttr=1):
    for i in range(epoch):
        dWG = np.asarray([np.zeros(w[ii].shape) for ii in range(len(w))])
        dBG = np.asarray([np.zeros(b[ii].shape) for ii in range(len(b))])
        if i == 0:
            dW = []
            dB = []

        # for j in range(len(trainingSet)):
        trainData = trainingSet
        # print(trainData)
        inpu = np.delete(trainData, -resAttr, 1)
        inpu = pp.normalRange(inpu).tolist()
        # print(inpu)
        d = np.delete(trainData, range(len(trainData[0])-resAttr), 1)
        d = pp.normalRange(d)

        # print(inpu)
        
        o = mlp.feedForward(inpu, w, b, a)
        dW,dB = mlp.backpropagate(o,w,dW,dB,b,a,d,leanRate,momentum)
        dWG += np.asarray(dW)
        dBG += np.asarray(dB)
        w = dWG/len(trainingSet)
        b = dBG/len(trainingSet)
        # print(w)
        trainData = trainingSet
        inpu = np.delete(trainData, -1, 1)
        inpu = pp.normalRange(inpu).tolist()
        d = np.delete(trainData, range(len(trainData[0])-resAttr), 1)
        d = pp.normalRange(d)
        o = mlp.feedForward(inpu, w, b, a)
        print(i)
        if i % 1 == 0:
            print('error',mlp.mse(o,d))

    return w, b

def testing(w, b, a, data, resAttr=1):
    res = 0.0
    # print(data)
    inpu = np.delete(data, -resAttr, 1)
    inpu = pp.normalRange(inpu).tolist()
    d = np.delete(data, range(len(data[0])-resAttr), 1)
    # d = pp.normalRange(d)

    o = mlp.feedForward(inpu, w, b, a)[-1]
    print(o)
    o = np.round(pp.normalize(o, denorm=True))

    ans = np.sum(d - o,axis=1)

    for i in range(len(ans)):
        if ans[i] == 0:
            res += 1.0
    print(o)
    print(d)
    print(ans)
    return res/len(ans)
    
if __name__ == '__main__':
    data = pp.input('Flood_dataset.txt')
    trainSet, testSet = pp.kFolds(data,10)
    w,b,a = mlp.modelInit('8x-6s-1s')
    chunk = trainSet[0][:40]
    # print('chunk', chunk)
    inpu = np.delete(chunk, -1, 1)
    inpu = pp.normalRange(inpu).tolist()
    # print('inpu',inpu)
    d = np.delete(chunk, range(len(chunk[0])-1), 1)
    d = pp.normalRange(d)

    for i in range(50000):
        if i == 0:
            dW = []
            dB = []
        # print(i)
        # print('legit', inpu)
        # print('legit', d)
        # print('inpu',inpu)
        # print('w',w)
        o = mlp.feedForward(inpu, w, b, a)
        # print(o)
        # print('b',mlp.mse(o,d))
        # for i in o:
        #     print(i)
        # print('w',  w)
        w,b,dW,dB = mlp.backpropagate(inpu,o,w,dW,dB,b,a,d,0.001,0)
        # print('dW',dW)
        # print('w',w.__class__)
        # w += dW
        # print('w',w.__class__)
        # b += dB
        # print(dW)
        # print(w)
        # print(dB)
        o = mlp.feedForward(inpu, w, b, a)
        print('a',mlp.mse(o,d))
    print(np.round(pp.normalize(o[-1], denorm=True)))
    print(np.round(pp.normalize(d, denorm=True)))
    

    # nw,nb = training(trainSet[0],1,copy.deepcopy(w),copy.deepcopy(b),a,0.001,0.001,1)

    # print(len(testSet[0]))
    # print(testing(nw,nb,a,testSet[0]))
    # print(np.average(trainSet[0],axis=0))
    # print(w)
    # print(b)
    print()
    # print(nw)
    # print(nb)
