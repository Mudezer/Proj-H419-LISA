import os
import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import albumentations as A
from albumentations.pytorch import ToTensorV2

from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score, roc_curve, auc

from src.data import SegmentationDataset
from src.nnet.unet.UNET import UNET
from src.utils import (
    getTestSet,
    getDevice,
)

def compute_metrics(y, y_pred):
    '''
    Compute the metrics for the model
    :param y:  ground truth
    :param y_pred:  predictions
    :return:  accuracy, f1, precision, recall, roc_auc, roc_curve_fpr, roc_curve_tpr
    '''

    """"Ground truth"""""
    y = y.cpu().detach().numpy()
    y = y > 0.5
    y = y.astype(np.uint8)
    y = y.reshape(-1)

    """Predictions"""
    y_pred = y_pred.cpu().detach().numpy()
    y_pred = y_pred > 0.5
    y_pred = y_pred.astype(np.uint8)
    y_pred = y_pred.reshape(-1)

    accuracy = accuracy_score(y, y_pred)
    f1 = f1_score(y, y_pred)
    precision = precision_score(y, y_pred)
    recall = recall_score(y, y_pred)
    roc_auc = roc_auc_score(y, y_pred)
    roc_curve_fpr, roc_curve_tpr, _ = roc_curve(y, y_pred)

    return accuracy, f1, precision, recall, roc_auc, roc_curve_fpr, roc_curve_tpr

if __name__ == "__main__":


    test_folder = "../assets/test"
    test_masks = "../assets/test_masks"
    output_dir = "../output/unet/results/evaluation_3.csv"
    output_img = "../output/unet/predictions/unet_reduced_3/"


    # Load the data
    image_height = 320  # 320
    image_width = 480  # 480

    # Define the transformations
    test_transforms = A.Compose(
        [
            A.Resize(height=image_height, width=image_width), # Resize the image to a fixed size
            A.Normalize( # Normalize the image
                mean=[0.0, 0.0, 0.0], # mean and standard deviation for the three channels
                std=[1.0, 1.0, 1.0],
                max_pixel_value=255.0, # maximum pixel value
            ),
            ToTensorV2(), # Convert the image to a PyTorch tensor
        ]
    )

    num_workers = 2
    pin_memory = True
    LOAD_MODEL = False
    batch_size = 4  # 4

    # generate the test set data loader
    testloader = getTestSet(test_folder,
                            test_masks,
                            test_transforms,
                            batch_size,
                            num_workers,
                            pin_memory,
                            LOAD_MODEL)

    ### hyperparameters
    LEARNING_RATE = 1e-4
    device = getDevice()
    num_epochs = 3
    checkpoint_path = "../output/unet/state/my_checkpoint_3.pth"

    print("=> Loading model")
    # create the model
    model = UNET(in_channels=3, out_channels=1)
    model.to(device) # move the model to the device
    model.load_state_dict(torch.load(checkpoint_path, map_location=device)) # load the model state from the checkpoint
    model.eval() # set the model to evaluation mode

    # Loss and optimizer
    print("=> metrics")

    accuracy = []
    f1 = []
    precision = []
    recall = []
    roc_auc = []
    roc_curve_fpr = []
    roc_curve_tpr = []



    loop = tqdm(testloader)

    for idx, (x,y) in enumerate(loop):
        X = x.to(device)
        y = y.unsqueeze(1).to(device)
        with torch.no_grad():
            #
            preds = torch.sigmoid(model(X))

            # return accuracy, f1, precision, recall, roc_auc, roc_curve_fpr, roc_curve_tpr
            metrics = compute_metrics(y, preds)
            accuracy.append(metrics[0])
            f1.append(metrics[1])
            precision.append(metrics[2])
            recall.append(metrics[3])
            roc_auc.append(metrics[4])
            roc_curve_fpr.append(metrics[5])
            roc_curve_tpr.append(metrics[6])

            preds = (preds > 0.5).float()

            preds_ex = preds.expand(-1, 3, -1, -1)  # Expand the second dimension to 3
            y_ex = y.expand(-1, 3, -1, -1)  # Expand the second dimension to 3

            preds_ex = preds_ex.cpu()
            y_ex = y_ex.cpu()

            concatenated = torch.cat((x, y_ex, preds_ex), dim=2) # Concatenate the images
            torchvision.utils.save_image(concatenated, f"{output_img}test_{idx}.png") # Save the image

    # Save the metrics to a csv file
    df = pd.DataFrame(data={
        "accuracy": accuracy,
        "f1": f1,
        "precision": precision,
        "recall": recall,
        "roc_auc": roc_auc,
        "roc_curve_fpr": roc_curve_fpr,
        "roc_curve_tpr": roc_curve_tpr
    })
    df.to_csv(output_dir, sep=",", index=False)

    print("Done")

