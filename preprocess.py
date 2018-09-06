def input(input):
    res = []
    with open(input, 'r') as inputFile:
        res = [line.rstrip().split('\t') for line in inputFile]
    return res