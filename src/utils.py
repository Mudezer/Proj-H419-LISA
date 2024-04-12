import torch
import torchvision
from src.data import SegmentationDataset
from torch.utils.data import DataLoader
import pandas as pd
import random
import os
import numpy as np
import cv2
import tqdm
from glob import glob
from PIL import Image


def getLoaders(train_dir,
             train_masks,
             test_dir,
             test_masks,
             train_transforms,
             test_transforms,
             batch_size,
             num_workers,
             pin_memory,
             LOAD_MODEL):
    train_ds = SegmentationDataset(image_dir=train_dir,
                                mask_dir=train_masks,
                                transform=train_transforms)
    train_loader = DataLoader(train_ds,
                            batch_size=batch_size,
                            num_workers=num_workers,
                            pin_memory=pin_memory,
                            shuffle=True)
    test_ds = SegmentationDataset(image_dir=test_dir,
                                mask_dir=test_masks,
                                transform=test_transforms)
    test_loader = DataLoader(test_ds,
                            batch_size=batch_size,
                            num_workers=num_workers,
                            pin_memory=pin_memory,
                            shuffle=False)

    return train_loader, test_loader

def getTestSet(test_dir,
             test_masks,test_transforms,
             batch_size,
             num_workers,
             pin_memory,
             LOAD_MODEL):
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
    print("=> Saving checkpoint")
    torch.save(state, filename)

def load_checkpoint(checkpoint, model, optimizer):
    print("=> Loading checkpoint")
    model.load_state_dict(checkpoint["state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer"])

def output_loss_to_csv(epochs, train_loss,validation_loss, path = "../output/unet/results/training_losses.csv"):
    df = pd.DataFrame(data={"epoch":epochs,"train_loss":train_loss, "validation_loss":validation_loss})
    df.to_csv(path, sep=",",index=False)

""" Seeding the randomness. """
def set_seed(seed):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True

def mask_parse(mask):
    mask = np.expand_dims(mask, axis=-1)
    mask = np.concatenate([mask, mask, mask], axis=-1)
    return mask


def test(loader, model, folder=None, device = "mps"):
    model.eval()

    for idx, (x,y) in list(enumerate(loader))[:1]:
        X = x.to(device)
        y = y.to(device)
        with torch.no_grad():
            preds = torch.sigmoid(model(X))
            preds = (preds > 0.5).float()

        print(f"y before unsqueeze : {y.shape}")
        print(f"preds before expansion : {preds.shape}")
        yy = y.unsqueeze(1)

        print(preds.shape)
        print(y.shape)
        print(yy.shape)

        preds_ex = preds.expand(-1, 3, -1, -1)  # Expand the second dimension to 3
        yy_ex = yy.expand(-1, 3, -1, -1)  # Expand the second dimension to 3
        print(x.shape)
        print("new shapes")
        print(preds_ex.shape)
        print(yy_ex.shape)

        preds_ex = preds_ex.cpu()
        yy_ex = yy_ex.cpu()

        print(f"device for x: {x.get_device()}")
        print(f"device for preds: {preds_ex.get_device()}")
        print(f"device for yy: {yy_ex.get_device()}")



        concatenated_tensor = torch.cat((x, yy_ex, preds_ex), dim=2)
        torchvision.utils.save_image(concatenated_tensor,f"test_{idx}.png")






















