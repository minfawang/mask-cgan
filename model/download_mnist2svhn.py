import os
import torch
import torchvision
from torchvision import datasets, transforms
from PIL import Image


def get_mnist_loader():
    transform32 = transforms.Compose([
        transforms.Resize(32, Image.BICUBIC),
        transforms.ToTensor(),
    ])
    trainset = datasets.MNIST('datasets/mnist_train', train=True, download=True, transform=transform32)
    data_loader = torch.utils.data.DataLoader(trainset, batch_size=64, shuffle=False)
    return data_loader


def get_svhn_loader():
    trainset = datasets.SVHN('datasets/svhn_train', split='train', download=True, transform=transforms.ToTensor())
    data_loader = torch.utils.data.DataLoader(trainset, batch_size=64, shuffle=False)
    return data_loader


def save_images(data_loader, side, train_count, test_count):
    dataset = 'train'  # one of train | test
    index = 0
    max_count = train_count + test_count
    for images, labels in data_loader:
        if images.size(1) == 1:
            images = images.squeeze(1)  # [batch, 1, 28, 28] -> [batch, 28, 28]
        else:
            assert images.size(1) == 3  # [batch, 3, 32, 32]

        for image in images:
            torchvision.utils.save_image(image, f'datasets/mnist2svhn/{dataset}/{side}/{index:04d}.png')
            index += 1

            if index >= train_count:
                dataset = 'test'

            if index >= max_count:
                break
        if index >= max_count:
            break


os.makedirs('datasets/mnist2svhn/train/A', exist_ok=True)
os.makedirs('datasets/mnist2svhn/test/A', exist_ok=True)
os.makedirs('datasets/mnist2svhn/train/B', exist_ok=True)
os.makedirs('datasets/mnist2svhn/test/B', exist_ok=True)
train_count = 1000
test_count = 1000
save_images(get_mnist_loader(), 'A', train_count, test_count)
save_images(get_svhn_loader(), 'B', train_count, test_count)
