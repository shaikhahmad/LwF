import torch
from torchvision import datasets
import os
from PIL import Image

def pil_loader(path):
    # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
    with open(path, 'rb') as f:
        img = Image.open(f)
        return img.convert('RGB')

def find_classes(dir):
    classes = [d for d in os.listdir(dir) if os.path.isdir(os.path.join(dir, d))]
    classes.sort()
    class_to_idx = {classes[i]: i for i in range(len(classes))}
    return classes, class_to_idx

def default_loader(path):
    from torchvision import get_image_backend
    if get_image_backend() == 'accimage':
        return accimage_loader(path)
    else:
        return pil_loader(path)


class Scenes(datasets.ImageFolder):
    def __init__(self, root, transform=None, target_transform=None,
                 train=True):
        imagelist_file = 'Images.txt'
        if train:
            imagelist_file = 'Train' + imagelist_file
        else :
            imagelist_file = 'Test' + imagelist_file
        filesnames = open(os.path.join(root, imagelist_file)).read().splitlines()
        self.root = os.path.join(root, 'Images')
        classes, class_to_idx = find_classes(self.root)

        images = []

        for filename in list(set(filesnames)):
            target = filename.split('/')[0]
            path = os.path.join(root, 'Images/' + filename)
            item = (path, class_to_idx[target])
            images.append(item)

        self.classes = classes
        self.class_to_idx = class_to_idx
        self.samples = images

        self.imgs = self.samples
        self.loader = default_loader
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.samples)


class CUB(datasets.ImageFolder):
    def __init__(self, root, transform=None, target_transform=None,
                 train=True):
        if train:
            imagelist_file = 'train.txt'
        else :
            imagelist_file = 'test.txt'
        filesnames = open(os.path.join(root, imagelist_file)).read().splitlines()
        self.root = os.path.join(root, 'images')
        classes, class_to_idx = find_classes(self.root)

        images = []

        for filename in list(set(filesnames)):
            target = filename.split('/')[0]
            path = os.path.join(root, 'images/' + filename)
            item = (path, class_to_idx[target])
            images.append(item)

        self.classes = classes
        self.class_to_idx = class_to_idx
        self.samples = images

        self.imgs = self.samples
        self.loader = default_loader
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.samples)

class CIFAR10(datasets.CIFAR10):
    def __init__(self, root, extend=0, *args, **kwargs):
        self.extend = extend
        super().__init__(root, *args, **kwargs)
        self.targets = [x+extend for x in self.targets]

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        img, target = self.data[index], self.targets[index]

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        img = Image.fromarray(img)

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target

class MNIST(datasets.MNIST):
    def __init__(self, root, extend=0, *args, **kwargs):
        self.extend = extend
        super().__init__(root, *args, **kwargs)
        self.targets = [x + self.extend for x in self.targets]

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        img, target = self.data[index], int(self.targets[index])

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        img = Image.fromarray(img.numpy(), mode='L')

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target

class FashionMNIST(datasets.FashionMNIST):
    def __init__(self, root, extend=0, *args, **kwargs):
        self.extend = extend
        super().__init__(root, *args, **kwargs)
        self.targets = [x + self.extend for x in self.targets]

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        img, target = self.data[index], int(self.targets[index])

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        img = Image.fromarray(img.numpy(), mode='L')

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target
