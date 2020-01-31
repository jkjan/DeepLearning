import numpy as np
import sys, os
sys.path.append(os.pardir)
from mnist import load_mnist

(xTrain, tTrain), (xTest, tTest) = \
    load_mnist(normalize=True, one_hot_label=True)

print(xTrain.shape)
print(tTrain.shape)

trainSize = xTrain.shape[0]
batchSize = 10
batchMask = np.random.choice(trainSize, batchSize)  # 60000 개를 다 넣어볼 순 없으니 랜덤으로 고르자. 60000 개 중 10개를 고른다.
xBatch = xTrain[batchMask]
tBatch = tTrain[batchMask]

def crossEntropyError(y, t) :
    if y.dim == 1 :
        t = t.reshape(1, t.size)
        y = y.reshape(1, y.size)
    batchSize = y.shape[0]
    return -np.sum(np.log(y[np.arange(batchSize), t]+1e-7)) / batchSize
# y[np.arange(batchSize), t] 는 정답 레이블에 해당하는 신경망의 출력만을 뽑아온다.









