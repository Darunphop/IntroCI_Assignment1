import preprocess as pp
import MLP as mlp
import numpy as np
import sys
import copy
import matplotlib.pyplot as plt

def training(trainingSet, testSet, epoch, w, b, a, learnRate, momentum, resAttr=1):
    res = []
    inpu = np.delete(trainingSet, -resAttr, 1)
    inpu = pp.normalRange(inpu).tolist()
    d = np.delete(trainingSet, range(len(trainingSet[0])-resAttr), 1)
    d = pp.normalRange(d)

    for i in range(epoch):
            if i == 0:
                dW = []
                dB = []
            o = mlp.feedForward(inpu, w, b, a)
            if (i+1) % 100 == 0 or i == 0:
                # print(i+1, testing(w,b,a,testSet))
                res.append((i+1, testing(w,b,a,testSet)))
            w,b,dW,dB = mlp.backpropagate(inpu,o,w,dW,dB,b,a,d,learnRate,momentum)

    return np.asarray(res)

def testing(w, b, a, data, resAttr=1):
    inpu = np.delete(data, -resAttr, 1)
    inpu = pp.normalRange(inpu).tolist()
    d = np.delete(data, range(len(data[0])-resAttr), 1)


    o = mlp.feedForward(inpu, w, b, a)[-1]
    o = np.round(pp.normalize(o, denorm=True))

    return mlp.mse(o,d)
    
if __name__ == '__main__':
    if sys.argv[1] == 'exp1':
        inputFile = 'Flood_dataset.txt'
        model = '8x-5s-1x'
        epoch = 10000
        k = 10

        data = pp.input(inputFile)
        trainSet, testSet = pp.kFolds(data,k)
        res = []

        for i in range(k):
            w,b,a = mlp.modelInit(model)

            d1=training(trainSet[i],testSet[i],epoch,copy.deepcopy(w),copy.deepcopy(b),a,0.001,0.9)
            d2=training(trainSet[i],testSet[i],epoch,copy.deepcopy(w),copy.deepcopy(b),a,0.001,0.5)
            d3=training(trainSet[i],testSet[i],epoch,copy.deepcopy(w),copy.deepcopy(b),a,0.001,0.0)

            fig = plt.figure(i+1)
            plt.title('Fold '+str(i+1))
            plt.plot(d1[:,0], d1[:,1], 'bo-', label='l=0.001, m=0.9', ms=5)
            plt.plot(d2[:,0], d2[:,1], 'go-', label='l=0.001, m=0.5', ms=5)
            plt.plot(d3[:,0], d3[:,1], 'yo-', label='l=0.001, m=0.0', ms=5)
            plt.legend(loc='best')
            fig.savefig('exp1,'+str((i+1))+'.png')

            res.append([d1[:,1][-1],d2[:,1][-1],d3[:,1][-1]])
        

        with open("exp1_result.txt", "w") as outFile:
            outFile.write(str(np.asarray(res)))
        with open("exp1_result.txt", "a") as outFile:
            outFile.write('\n' + str(np.average(np.asarray(res), axis=0)))
    
    elif sys.argv[1] == 'exp2':
        pass
    else:
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
        
    
