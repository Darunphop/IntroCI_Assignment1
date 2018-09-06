import preprocess as pp

def modelInit(model):
    layerSize = [int(n) for n in model.split('-')]
    print(layerSize)

if __name__ == '__main__':
    data = pp.input('Flood_dataset.txt')
    trainSet, testSet = pp.kFolds(data,10)
    modelInit('8-2-6-1')
    # print(len(trainSet))
    # print('Hola')