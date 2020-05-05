from torch.autograd import Variable
import torch
import time
import math

def accuracy_test(model, data, device):
    correct = 0
    total = 0

    for i, [imgs, labels] in enumerate(data):
        img = Variable(imgs).to(device)
        label = Variable(labels).to(device)
        output = model(img)
        _, output_index = torch.max(output, 1)

        total += label.size(0)
        correct += (output_index == label).sum().float()

    print("Accuracy of Test Data : %.2f%%" % (100*correct/total))


def init():
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print("cuda is available!\n")
        torch.backends.cudnn.benchmark = True
        print('Memory Usage:')
        print('Max Alloc:', round(torch.cuda.max_memory_allocated(0) / 1024 ** 3, 1), 'GB')
        print('Allocated:', round(torch.cuda.memory_allocated(0) / 1024 ** 3, 1), 'GB')
        print('Cached:   ', round(torch.cuda.memory_cached(0) / 1024 ** 3, 1), 'GB')
        print('cuDNN:    ', torch.backends.cudnn.version())
        print()

    else:
        device = torch.device("cpu")

    return device


def time_since(since):
    now = time.time()
    s = now - since
    m = math.floor(s / 60)
    s -= m * 60
    return "%dm %ds" % (m, s)


