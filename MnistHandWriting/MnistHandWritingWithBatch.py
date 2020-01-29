import MnistHandWriting as mh
import numpy as np

x, t = mh.getData()
network = mh.initNetwork()

batchSize = 100
accuracyCnt = 0

for i in range(0, len(x), batchSize):
    xBatch = x[i:i+batchSize]   # 100개씩 끊어서
    yBatch = mh.predict(network, xBatch)
    p = np.argmax(yBatch, axis = 1)
    accuracyCnt += np.sum(p == t[i:i+batchSize])

print("Accuracy : " + str(float(accuracyCnt) / len(x)))