import os
import torch
#import cPickle as pickle
#import _pickle as cPickle
import numpy
import sys

cwd = os.getcwd()
sys.path.append(cwd+'/../')
import datasets as datasets
import datasets.transforms as transforms



input_size=227

def build_test_dataset(root='/data/lmdb_imagenet', batch_size=250, shuffle=False, num_workers=2):
    print(root+'/imagenet_mean.binaryproto')
    normalize = transforms.Normalize(
            meanfile=root+'/imagenet_mean.binaryproto')

    val_loader = torch.utils.data.DataLoader(
        datasets.ImageFolder(root, transforms.Compose([
            transforms.ToTensor(),
            normalize,
            transforms.CenterCrop(input_size),
        ]),
        Train=False),
        batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=False)
    return val_loader



def build_train_dataset(mode, example_weight, root='/data/lmdb_imagenet', batch_size=256, shuffle=False, num_workers=12):
    normalize = transforms.Normalize(
            meanfile=root+'/imagenet_mean.binaryproto')

    trainset = datasets.ImageFolder(
        root,
        transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
            transforms.RandomSizedCrop(input_size),
        ]),
        Train=True, example_weight=example_weight)
    #print(trainset.__dict__.keys())
    #normal train loader
    if mode == 'normal':
        # without weight
        return torch.utils.data.DataLoader(trainset, batch_size=batch_size,\
            shuffle=shuffle, num_workers=num_workers, pin_memory=False)
    elif mode == 'sample_batch':
        # sample batch by weights
        sampler = torch.utils.data.sampler.WeightedRandomSampler(example_weight.double().t()[0], len(trainset))
        return torch.utils.data.DataLoader(trainset, batch_size=batch_size, \
        shuffle=shuffle, num_workers=num_workers, \
        sampler=sampler, pin_memory=False)
    elif mode == 'sample_dataset':
        # sample from indecies of subset without replacement
        #indices = numpy.random.choice(len(trainset), size=len(trainset))
        indices = example_weight
        sampler = torch.utils.data.sampler.SubsetRandomSampler(indices)
        return torch.utils.data.DataLoader(trainset, batch_size=batch_size, \
        shuffle=shuffle, num_workers=num_workers, \
        sampler=sampler, pin_memory=False)
    else:
        raise Exception("Undefined Mode!", mode)
