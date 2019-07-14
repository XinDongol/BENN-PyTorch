import os
import torch
#import cPickle as pickle
import pickle as cPickle
import numpy
import torchvision.transforms as transforms

class dataset():
    def __init__(self, root=None, train=True, example_weight=None):
        self.root = root
        self.train = train
        self.transform = transforms.ToTensor()
        if self.train:
            train_data_path = os.path.join(root, 'train_data')
            #print(train_data_path)
            train_labels_path = os.path.join(root, 'train_labels')
            self.train_data = numpy.load(open(train_data_path, 'rb'))
            self.train_data = torch.from_numpy(self.train_data.astype('float32'))
            self.train_labels = numpy.load(open(train_labels_path, 'rb')).astype('int')
            self.example_weight = example_weight
        else:
            test_data_path = os.path.join(root, 'test_data')
            test_labels_path = os.path.join(root, 'test_labels')
            self.test_data = numpy.load(open(test_data_path, 'rb'))
            self.test_data = torch.from_numpy(self.test_data.astype('float32'))
            self.test_labels = numpy.load(open(test_labels_path, 'rb')).astype('int')

    def __len__(self):
        if self.train:
            return len(self.train_data)
        else:
            return len(self.test_data)

    def __getitem__(self, index):
        if self.train:
            img, target, weight = self.train_data[index], self.train_labels[index], self.example_weight[index]
        else:
            img, target, weight = self.test_data[index], self.test_labels[index], []
        
        return img, target, weight


def build_test_dataset(root='./data/', batch_size=1000, shuffle=False, num_workers=4):
    testset = dataset(root=root, train=False)
    return torch.utils.data.DataLoader(testset, batch_size=batch_size,\
    shuffle=shuffle, num_workers=num_workers)

def build_train_dataset(mode, example_weight, root='./data/', batch_size=128, shuffle=False, num_workers=4):
    trainset = dataset(root=root, train=True, example_weight=example_weight)
    #normal train loader
    if mode == 'normal':
        # without weight
        return torch.utils.data.DataLoader(trainset, batch_size=batch_size,\
            shuffle=shuffle, num_workers=num_workers)
    elif mode == 'sample_batch':
        # sample batch by weights
        sampler = torch.utils.data.sampler.WeightedRandomSampler(example_weight.double(), len(trainset))
        return torch.utils.data.DataLoader(trainset, batch_size=batch_size, \
        shuffle=shuffle, num_workers=num_workers, \
        sampler=sampler)
    elif mode == 'sample_dataset':
        # sample from indecies of subset without replacement
        #indices = numpy.random.choice(len(trainset), size=len(trainset))
        indices = example_weight
        sampler = torch.utils.data.sampler.SubsetRandomSampler(indices)
        return torch.utils.data.DataLoader(trainset, batch_size=batch_size, \
        shuffle=shuffle, num_workers=num_workers, \
        sampler=sampler)
    else:
        raise Exception("Undefined Mode!", mode)














