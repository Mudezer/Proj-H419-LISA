import os
import numpy as np
import pandas as pd
import tqdm as tqdm
import torch
import torch.nn as nn
import torch.optim as optim
import albumentations as A
from albumentations.pytorch import ToTensorV2

from sklearn.metrics import accuracy_score, f1_score

from src.utils import (
    getLoaders,
    getDevice,
    save_checkpoint,
    output_loss_to_csv,
    set_seed,
    # save_pred_to_img,

)


from src.nnet.unet.UNET import UNET


def train(model, loader, optimizer, loss_fn, device):
    epoch_loss = 0.0
    loop = tqdm.tqdm(loader)

    accuracies = []
    f1_scores = []

    model = model.train()

    for batch_idx, (x_train, y_train) in enumerate(loop):
        x_train = x_train.to(device=device)
        y_train = y_train.float().unsqueeze(1).to(device=device)

        y_pred = model(x_train)
        accuracy, f1 = compute_metrics(y_train, y_pred)
        accuracies.append(accuracy)
        f1_scores.append(f1)
        loss = loss_fn(y_pred, y_train)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        loop.set_postfix(loss=loss.item())
        epoch_loss += loss.item()

    accuracies = np.mean(accuracies)
    f1_scores = np.mean(f1_scores)
    epoch_loss = epoch_loss/len(loader)
    return epoch_loss, accuracies, f1_scores


def evaluate(model, loader, loss_fn, device="mps"):
    epoch_loss = 0.0
    num_correct = 0
    num_pixels = 0
    dice_score = 0

    accuracies = []
    f1_scores = []
    model.eval()
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            y = y.float().unsqueeze(1).to(device=device)

            y_hat = model(x)
            ## compute loss
            loss = loss_fn(y_hat, y)

            y_pred = torch.sigmoid(y_hat)

            ## compute metrics + metrics retrieval
            accuracy, f1 = compute_metrics(y, y_pred)
            accuracies.append(accuracy)
            f1_scores.append(f1)

            y_pred = (y_pred > 0.5).float()
            num_correct += (y_pred == y).sum()
            num_pixels += torch.numel(y)
            dice_score += (2 * (y * y_pred).sum()) / (
                (y + y_pred).sum() + 1e-8
            )
            epoch_loss += loss.item()

        epoch_loss = epoch_loss/len(loader)
        accuracies = np.mean(accuracies)
        f1_scores = np.mean(f1_scores)



    print(f"Got {num_correct}/{num_pixels} with accuracy {num_correct/num_pixels*100:.2f}")
    print(f"Dice score: {dice_score/len(loader)}")
    model.train()

    return epoch_loss, accuracies, f1_scores, (dice_score/len(loader)).cpu().detach().numpy()

def compute_metrics(y, y_pred):
    """ground truth"""""
    y = y.cpu().detach().numpy()
    y = y > 0.5
    y = y.astype(np.uint8)
    y = y.reshape(-1)
    """"masks"""
    y_pred = y_pred.cpu().detach().numpy()
    y_pred = y_pred > 0.5
    y_pred = y_pred.astype(np.uint8)
    y_pred = y_pred.reshape(-1)

    accuracy = accuracy_score(y, y_pred)
    f1 = f1_score(y, y_pred)
    return accuracy, f1


def main():

    # Load the data
    image_height =  320 #160
    image_width = 480 #240


    train_transforms = A.Compose(
        [
            A.Resize(height=image_height, width=image_width),
            A.Rotate(limit=35, p=1.0),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.1),
            A.Normalize(
                mean=[0.0, 0.0, 0.0],
                std=[1.0, 1.0, 1.0],
                max_pixel_value=255.0,
            ),
            ToTensorV2(),
        ]
    )

    test_transforms = A.Compose(
        [
            A.Resize(height=image_height, width=image_width),
            A.Normalize(
                mean=[0.0, 0.0, 0.0],
                std=[1.0, 1.0, 1.0],
                max_pixel_value=255.0,
            ),
            ToTensorV2(),
        ]
    )

    train_dir = "../assets/train/"
    train_masks = "../assets/train_masks/"
    test_dir = "../assets/validation/"
    test_masks = "../assets/validation_masks/"
    checkpoint_path = "../output/unet/state/my_checkpoint_2.pth"
    results = "../output/unet/results/training_losses_2.csv"

    num_workers = 2
    pin_memory  = True
    LOAD_MODEL = False
    batch_size = 4 # 4

    trainloader, testloader = getLoaders(train_dir,
                                         train_masks,
                                         test_dir,
                                         test_masks,
                                         train_transforms,
                                         test_transforms,
                                         batch_size,
                                         num_workers,
                                         pin_memory,
                                         LOAD_MODEL)

    # Hyperparameters
    LEARNING_RATE = 1e-4
    lr = .1
    lr2 = .01
    lr3 = .001
    lr4 = 10

    DEVICE = getDevice()
    num_epochs = 10


    device = torch.device(DEVICE)
    model = UNET(in_channels=3, out_channels=1)
    model = model.to(device)


    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    loss_fn = nn.BCEWithLogitsLoss()  # cross entropy loss for multiple classes

    ## for retrieving data and getting performance overview
    epochs = []
    train_loss_tot = []
    valid_loss_tot = []
    accuracies_val = []
    f1_scores_val = []
    accuracies_train = []
    f1_scores_train = []
    Dice = []

    best_valid_loss = float("inf")

    for epoch in range(num_epochs):
        print(f"Epoch {epoch+1}")
        epochs.append(epoch+1)
        train_loss, accuracy_train, f1_train = train(model, trainloader, optimizer, loss_fn, device)
        train_loss_tot.append(train_loss)
        accuracies_train.append(accuracy_train)
        f1_scores_train.append(f1_train)
        valid_loss, accuracy_val, f1_val, dice = evaluate(model, testloader, loss_fn, device)
        valid_loss_tot.append(valid_loss)
        accuracies_val.append(accuracy_val)
        f1_scores_val.append(f1_val)
        Dice.append(dice)


        if valid_loss < best_valid_loss:
            data = f"Validation loss improved from {best_valid_loss:2.4f} to {valid_loss:2.4f}. \n Saving checkpoint: {checkpoint_path}"
            print(data)
            # checkpoint = {
            #     "state_dict": model.state_dict(),
            #     "optimizer": optimizer.state_dict(),
            # }
            best_valid_loss = valid_loss

            save_checkpoint(model.state_dict(), checkpoint_path)


    df = pd.DataFrame(data={"epoch": epochs,
                            "train_loss": train_loss_tot,
                            "validation_loss": valid_loss_tot,
                            "accuracy_train": accuracies_train,
                            "f1_train": f1_scores_train,
                            "accuracy_val": accuracies_val,
                            "f1_val": f1_scores_val,
                            "dice_val": Dice
                            })
    df.to_csv(results, sep=",", index=False)



if __name__ == "__main__":
    main()
