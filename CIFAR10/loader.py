import torchvision
import torchvision.transforms as tf
import torch

def load_data(batch_size):
    transform_train = tf.Compose([
        tf.RandomHorizontalFlip(),
        tf.RandomRotation(10),
        tf.RandomAffine(0, shear=10, scale=(0.8, 1.2)),
        tf.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        tf.ToTensor(),
        tf.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    transform_test = tf.Compose([
        tf.ToTensor(),
        tf.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    cifar_train = torchvision.datasets.CIFAR10(root="./data", train=True,
                                            download=True, transform=transform_train,
                                            target_transform=None)

    cifar_test = torchvision.datasets.CIFAR10(root="./root", train=False,
                                       download=True, transform=transform_test, target_transform=None)

    train_loader = torch.utils.data.DataLoader(list(cifar_train)[:], batch_size=batch_size, shuffle=True, num_workers=0, drop_last=True)
    test_loader = torch.utils.data.DataLoader(cifar_test, batch_size=batch_size, shuffle=False, num_workers=0, drop_last=True)

    return train_loader, test_loader