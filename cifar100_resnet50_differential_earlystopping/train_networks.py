import torchvision
import torch

# Merge the train and testing sets to allow for sampling
cifar100_train  = torchvision.datasets.CIFAR100("../cifar100_data", train=True, download=True)
cifar100_test   = torchvision.datasets.CIFAR100("../cifar100_data", train=False, download=True)

cifar100 = torch.utils.data.ConcatDataset([cifar100_train, cifar100_test])
