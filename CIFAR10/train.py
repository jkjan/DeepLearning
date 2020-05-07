from loader import load_data
import torch.nn as nn
from CNN import CNN
from utils import *
import torch
from torch.autograd import Variable
from torch.optim import lr_scheduler
import sys
import time
import matplotlib.pyplot as plt

# hyper parameters
batch_size = 256
lr = 0.1
n_iter = 100

# loss check
print_per = 10

# initiating
device = init()

# model declaring
model = CNN(batch_size).to(device)

# loss function
criterion = nn.CrossEntropyLoss()

# optimizer
# optimizer = torch.optim.SGD(model.parameters(), lr=lr,
#                       momentum=0.9, weight_decay=5e-4)
optimizer = torch.optim.Adam(model.parameters(), lr=lr)

# scheduler
scheduler = lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)

# to train
model.train()

# time when training starts
start = time.time()

sys.stdout.write("loading data...\n")

# data loading
cifar_train, cifar_test = load_data(batch_size)

sys.stdout.write("\ntraining start! total %d batches in a training set.\n\n" % len(cifar_train))

# loss tracking
losses = []
cur_loss = 0

for i in range(n_iter):
    loss = None
    for j, [image, label] in enumerate(cifar_train):
        input = Variable(image).to(device)
        target = Variable(label).to(device)

        optimizer.zero_grad()
        output = model(input)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()

        cur_loss += loss.item()


    if i % print_per == 0 and i != 0:
        sys.stdout.write("%d %d%% (%s) %.4f\n" % (i, i / n_iter * 100, time_since(start), cur_loss/len(cifar_train)/print_per))
        losses.append(cur_loss/len(cifar_train)/print_per)
        cur_loss = 0


# loss graph
plt.figure()
plt.plot(losses)
plt.show()

# to test
model.eval()

# testing accuracy
accuracy_test(model, cifar_test, device)

# save parameters
torch.save(model, "cifar10_cnn.pkl")