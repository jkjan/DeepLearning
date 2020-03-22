import torch
import torchvision
import torchvision.transforms as tf
from show import *

transform = tf.Compose(
    [tf.ToTensor(),
    tf.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

trainSet = torchvision.datasets.CIFAR10(root="./data", train=True,
                                        download=True, transform=transform)

trainLoader = torch.utils.data.DataLoader(trainSet, batch_size=4, shuffle=True, num_workers=0)

testSet = torchvision.datasets.CIFAR10(root="./root", train=False,
                                       download=True, transform=transform)

testLoader = torch.utils.data.DataLoader(testSet, batch_size=4, shuffle=False, num_workers=0)

classes = ('plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

dataIter = iter(trainLoader)
images, labels = dataIter.next()
print(images, labels)
showImg(torchvision.utils.make_grid(images))
print(' '.join('%5s' % classes[labels[j]] for j in range(4)))
