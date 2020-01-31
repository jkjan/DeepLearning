from commonlyused import *
from mnist import load_mnist

class TwoLayerNet:
    def __init__(self, inputSize, hiddenSize, outputSize, weightInitStd = 0.01):
        self.params = {'W1': weightInitStd * np.random.randn(inputSize, hiddenSize),
                       'b1': np.zeros(hiddenSize),
                       'W2': weightInitStd * np.random.randn(hiddenSize, outputSize),
                       'b2': np.zeros(outputSize)}

    def predict(self, x):
        W1, W2 = self.params['W1'], self.params['W2']
        b1, b2 = self.params['b1'], self.params['b2']
        a1 = np.dot(x, W1) + b1
        z1 = sigmoid(a1)
        a2 = np.dot(z1, W2) + b2
        return softmax(a2)

    def loss(self, x, t):
        y = self.predict(x)
        return crossEntropyError(y, t)

    def accuracy(self, x, t):
        y = self.predict(x)
        y = np.argmax(y, axis=1)
        t = np.argmax(t, axis=1)
        accuracy = np.sum(y == t) / float(x.shape[0])
        return accuracy

    def numericalGradient(self, x, t):
        lossW = lambda W : self.loss(x, t)
        grads = {'W1': numericalGradient(lossW, self.params['W1']),
                 'b1': numericalGradient(lossW, self.params['b1']),
                 'W2': numericalGradient(lossW, self.params['W2']),
                 'b2': numericalGradient(lossW, self.params['b2'])}
        return grads

print("hello")
(xTrain, tTrain), (xTest, tTest) = \
    load_mnist(normalize=True, one_hot_label=True)

network = TwoLayerNet(inputSize=784, hiddenSize=50, outputSize=10)

itersNum = 10 # 반복 횟수
trainSize = xTrain.shape[0] # 한 훈련의 크기
batchSize = 100 # 배치 크기
learningRate = 0.1

trainLossList = []
trainAccList = []
testAccList = []

iterPerEpoch = max(trainSize/batchSize, 1)

for i in range(itersNum):
    batchMask = np.random.choice(trainSize, batchSize)
    xBatch = xTrain[batchMask]
    tBatch = tTrain[batchMask]

    grad = network.numericalGradient(xBatch, tBatch)

    for key in ('W1', 'b1', 'W2', 'b2'):
        network.params[key] -= learningRate * grad[key]

    loss = network.loss(xBatch, tBatch)
    trainLossList.append(loss)

    if i % iterPerEpoch == 0:
        trainAcc = network.accuracy(xTrain, tTrain)
        testAcc = network.accuracy(xTest, tTest)
        trainAccList.append(trainAcc)
        testAccList.append(testAcc)
        print("train acc, test acc | " + str(trainAcc) + ", " + str(testAcc))