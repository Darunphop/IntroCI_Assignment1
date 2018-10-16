import preprocess as pp
import MLP as mlp
import numpy as np

def training(dataChunk, epoch):
    res = []

    return res
    
if __name__ == '__main__':
    data = pp.input('Flood_dataset.txt')
    trainSet, testSet = pp.kFolds(data,10)
    w,b,a = mlp.modelInit('8x-3s-2s-5s-2s')
    chunk = trainSet[0][:2]
    inpu = np.delete(chunk, -1, 1)
    inpu = pp.normalRange(inpu).tolist()
    d = np.delete(chunk, range(len(chunk[0])-2), 1)
    d = pp.normalRange(d).tolist()

    
    o = mlp.feedForward(inpu, w, b, a)
    # print(o)
    print('b',mlp.mse(o,d))
    # for i in o:
    #     print(i)
    # print(w)
    w,b,dW,dB = mlp.backpropagate(o,w,[],[],b,a,d,0.1,0)
    # print(w)
    # print(dB)
    o = mlp.feedForward(inpu, w, b, a)
    print('a',mlp.mse(o,d))
    # print(len(trainSet[0]))
    # print(trainSet[0][:1])
    # print(act.sigmoid([1,2,3]))
    # print(a)