import preprocess as pp
import MLP as mlp

if __name__ == '__main__':
    data = pp.input('Flood_dataset.txt')
    trainSet, testSet = pp.kFolds(data,10)
    w,b,a = mlp.modelInit('9x-3s-2s-5s-1s')
    inpu = trainSet[0][:2]
    inpu = pp.normalRange(inpu).tolist()
    # print(inpu.__class__)
    # print(pp.normalRange(inpu).__class__)
    o = mlp.feedForward(inpu, w, b, a)
    print(o)
    # print(len(trainSet[0]))
    # print(trainSet[0][:1])
    # print(act.sigmoid([1,2,3]))
    # print(a)