import torch
import torchvision.datasets as ds
import torchvision.transforms as tf

def download_train_and_test():
    mnist_train = ds.MNIST(root="mnist_dataset/",
                           train=True,
                           transform=tf.ToTensor(),
                           download=True)

    mnist_test = ds.MNIST(root="mnist_dataset/",
                          train=False,
                          transform=tf.ToTensor(),
                          download=True)

    return mnist_train, mnist_test

def data_loader(mnist_train, mnist_test, batch_size):
    return torch.utils.data.DataLoader(dataset=mnist_train,
                                batch_size=batch_size,
                                shuffle=True,
                                drop_last=True)
