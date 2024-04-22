import torch
import torchvision
from src.data import SegmentationDataset
from torch.utils.data import DataLoader
import pandas as pd
import random
import os
import numpy as np



def getLoaders(train_dir,# directory where the training images are stored
             train_masks, # directory where the training masks are stored
             test_dir, # directory where the testing images are stored
             test_masks, # directory where the testing masks are stored
             train_transforms, # transformations to be applied to the training images and masks
             test_transforms, # transformations to be applied to the testing images and masks
             batch_size, # number of samples per batch
             num_workers, # number of subprocesses to use for data loading
             pin_memory, # if True, the data loader will copy Tensors into CUDA pinned memory before returning them
             LOAD_MODEL):
    '''
    Set up the data loaders for the training and validating sets.
    1) Create the training and validation datasets
    2) Create the training and validation data loaders with the specified batch size, number of workers and pin memory
    3) Return the training and validation data loaders
    '''

    train_ds = SegmentationDataset(image_dir=train_dir,
                                mask_dir=train_masks,
                                transform=train_transforms)
    train_loader = DataLoader(train_ds,
                            batch_size=batch_size,
                            num_workers=num_workers,
                            pin_memory=pin_memory,
                            shuffle=True) # shuffle the data to avoid overfitting
    test_ds = SegmentationDataset(image_dir=test_dir,
                                mask_dir=test_masks,
                                transform=test_transforms)
    test_loader = DataLoader(test_ds,
                            batch_size=batch_size,
                            num_workers=num_workers,
                            pin_memory=pin_memory,
                            shuffle=False) # no need to shuffle the data for validation

    return train_loader, test_loader

def getTestSet(test_dir,
             test_masks,test_transforms,
             batch_size,
             num_workers,
             pin_memory,
             LOAD_MODEL):

    '''
    Set up the data loaders for the testing set.
    1) Create the testing dataset
    2) Create the testing data loader with the specified batch size, number of workers and pin memory
    3) Return the testing data loader
    '''

    test_ds = SegmentationDataset(image_dir=test_dir,
                                  mask_dir=test_masks,
                                  transform=test_transforms)
    test_loader = DataLoader(test_ds,
                             batch_size=batch_size,
                             num_workers=num_workers,
                             pin_memory=pin_memory,
                             shuffle=False)

    return test_loader

def getDevice():

    ''''
    Get the device to be used for the neural network.
    1) Check if a GPU is available,
    2) elif not check if the MPS is available,
    3) elif not use the CPU.
    '''
    device = (
        "cuda"
        if torch.cuda.is_available()
        else "mps"
        if torch.backends.mps.is_available()
        else "cpu"
    )

    print(f"Using {device} device")
    return device


def save_checkpoint(state, filename="../ouput/unet/state/my_checkpoint.pth"):
    '''
    Save the model state and optimizer state to a file.
    :param state: the model state and optimizer state
    :param filename: the path to the file where the model state and optimizer state will be saved
    :return: None
    '''
    print("=> Saving checkpoint")
    torch.save(state, filename)




















