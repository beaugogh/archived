import numpy as np

# save the csv train and test data into binary files
# for faster loading
def saveDataToBinary():
    train = np.genfromtxt('data/train.csv', delimiter=',')
    np.save('data/train', train)
    test = np.genfromtxt('data/test.csv', delimiter=',')
    np.save('data/test', test)
