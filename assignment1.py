


def input(input):
    res = []
    with open(input, 'r') as inputFile:
        res = [line.rstrip().split('\t') for line in inputFile]
    return res

if __name__ == '__main__':
    data = input('Flood_dataset.txt')
    print(data)
    print('Hola')