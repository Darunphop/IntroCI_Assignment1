import preprocess as pp
import MLP as mlp
import numpy as np

if __name__ == '__main__':
    data = pp.input('Flood_dataset.txt')
    trainSet, testSet = pp.kFolds(data,10)
    w,b,a = mlp.modelInit('8x-3s-2s-5s-1s')
    chunk = trainSet[0][:2]
    inpu = np.delete(chunk, -1, 1)
    inpu = pp.normalRange(inpu).tolist()
    d = np.delete(chunk, range(len(chunk[0])-1), 1)
    d = pp.normalRange(d).tolist()

    o = mlp.feedForward(inpu, w, b, a)
    # print(o)
    # for i in o:
    #     print(i)
    mlp.backpropagate(o,w,b,a,d,0.001,0)
    # print(len(trainSet[0]))
    # print(trainSet[0][:1])
    # print(act.sigmoid([1,2,3]))
    # print(a)