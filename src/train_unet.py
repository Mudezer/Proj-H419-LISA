import os
import numpy as np
import pandas as pd
import tqdm as tqdm
import torch
import torch.nn as nn
import torch.optim as optim
import albumentations as A # image augmentation library
from albumentations.pytorch import ToTensorV2

from sklearn.metrics import accuracy_score, f1_score

from src.utils import (
    getLoaders,
    getDevice,
    save_checkpoint,
)


from src.nnet.unet.UNET import UNET


def train(model, loader, optimizer, loss_fn, device):
    '''
    Train the model for one epoch.
    :param model:  the model to be trained
    :param loader:  the data loader (training set data loader)
    :param optimizer:  the optimizer used to update the model's weights
    :param loss_fn:  the loss function used to compute the loss
    :param device:  the device (CPU or GPU) used to run the model (mps in this case)
    :return:  the epoch loss, accuracy and f1 score for the training set
    '''
    epoch_loss = 0.0
    loop = tqdm.tqdm(loader)

    accuracies = []
    f1_scores = []

    model = model.train() # set the model to training mode

    for batch_idx, (x_train, y_train) in enumerate(loop):
        x_train = x_train.to(device=device)
        y_train = y_train.float().unsqueeze(1).to(device=device)

        y_pred = model(x_train)
        # compute the accuracy and f1 score
        accuracy, f1 = compute_metrics(y_train, y_pred)
        # append the accuracy and f1 score to the list
        accuracies.append(accuracy)
        f1_scores.append(f1)
        loss = loss_fn(y_pred, y_train)

        optimizer.zero_grad() # clear old gradient from last step (otherwise they would accumulate)
        loss.backward() # performs backprogapation to compute the gradient of the loss function w.r.t the model parameters
        optimizer.step() # updates the model parameters based on the gradient computed in the previous step

        loop.set_postfix(loss=loss.item()) # display the loss in the progress bar
        epoch_loss += loss.item() # sum the loss for the epoch to keep track of the total loss for the epoch

    accuracies = np.mean(accuracies) # compute the mean accuracy on the epoch
    f1_scores = np.mean(f1_scores) # compute the mean f1 score on the epoch
    epoch_loss = epoch_loss/len(loader) # compute the mean loss on the epoch
    return epoch_loss, accuracies, f1_scores


def evaluate(model, loader, loss_fn, device="mps"):
    '''
    :param model:
    :param loader:
    :param loss_fn:
    :param device:
    :return:
    '''
    epoch_loss = 0.0
    num_correct = 0
    num_pixels = 0
    dice_score = 0

    accuracies = []
    f1_scores = []
    model.eval() # set the model to evaluation mode
    with torch.no_grad(): # disable gradient computation to speed up the computation and reduce memory usage
        for x, y in loader: # iterate over the data loader
            x = x.to(device)
            y = y.float().unsqueeze(1).to(device=device)

            y_hat = model(x)

            ## compute loss between prediction and ground truth
            loss = loss_fn(y_hat, y)

            # apply sigmoid to the output of the model to map the output to the range [0,1]
            y_pred = torch.sigmoid(y_hat)

            ## compute metrics + metrics retrieval
            accuracy, f1 = compute_metrics(y, y_pred)
            accuracies.append(accuracy)
            f1_scores.append(f1)

            y_pred = (y_pred > 0.5).float() # binarizes the predictions based on a threshold of 0.5 (1 if > 0.5, 0 otherwise)
            num_correct += (y_pred == y).sum() # sum the number of correct predictions
            num_pixels += torch.numel(y) # count the number of pixels in the mask
            dice_score += (2 * (y * y_pred).sum()) / (
                (y + y_pred).sum() + 1e-8
            ) # compute the dice score also known as the F1 score
            epoch_loss += loss.item()

        epoch_loss = epoch_loss/len(loader)
        accuracies = np.mean(accuracies)
        f1_scores = np.mean(f1_scores)



    print(f"Got {num_correct}/{num_pixels} with accuracy {num_correct/num_pixels*100:.2f}")
    print(f"Dice score: {dice_score/len(loader)}")
    model.train() # set the model back to training mode

    return epoch_loss, accuracies, f1_scores, (dice_score/len(loader)).cpu().detach().numpy()

def compute_metrics(y, y_pred):
    '''
    Compute the accuracy and f1 score for the model.
    :param y:  the mask (ground truth)
    :param y_pred:  the predicted mask
    :return:  the accuracy and f1 score
    '''

    """ground truth"""""
    # detach the tensor from the computation graph and convert it to a numpy array
    # while switching it from the GPU to the CPU
    y = y.cpu().detach().numpy()
    y = y > 0.5 # threshold the pixel values to 0 or 1
    y = y.astype(np.uint8) # convert to 8-bit unsigned integer
    y = y.reshape(-1)
    """"masks"""
    y_pred = y_pred.cpu().detach().numpy()
    y_pred = y_pred > 0.5
    y_pred = y_pred.astype(np.uint8)
    y_pred = y_pred.reshape(-1)

    # use the sklearn metrics to compute the accuracy and f1 score
    accuracy = accuracy_score(y, y_pred)
    f1 = f1_score(y, y_pred)
    return accuracy, f1


def main():

    # Load the data
    image_height =  320 #160
    image_width = 480 #240


    train_transforms = A.Compose(
        [
            # resize the image to the desired size (in my case its mandatory as my machine could not handle the original size)
            A.Resize(height=image_height, width=image_width),
            # rotate the image by a random degree between -35 and 35 with a probability of 1.0
            A.Rotate(limit=35, p=1.0),
            # flips the image horizontally with a probability of 0.5
            A.HorizontalFlip(p=0.5), # classic augmentation technique as horizontal orientation does not change the meaning of the image
            # flips the image vertically with a probability of 0.1
            A.VerticalFlip(p=0.1), # like previous, can help in learning invariance to orientation
            # normalize the pixel values, mean and std are set to 0.0 and 1.0 respectively scaling the pixel value to the range [0;1]
            A.Normalize( # allows to speed up training by ensuring that the input features are on a similar scale
                mean=[0.0, 0.0, 0.0],
                std=[1.0, 1.0, 1.0],
                max_pixel_value=255.0,
            ),
            ToTensorV2(), # from numpy to pytorch tensors (HWC to CHW)
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

    # generate the data loaders for the training and testing set
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
    num_epochs = 10 # number of epochs to train the model

    # Differents features values for the model architecture
    feature1 = [64, 128, 256, 512]
    # feature2 = [32, 64, 128, 256]
    # feature3 = [16, 32, 64, 128]



    device = torch.device(DEVICE) # get the device (CPU or GPU)
    model = UNET(in_channels=3, out_channels=1, features=feature1) # create the UNET model
    model = model.to(device) # move the model to the device (CPU or GPU)

    # Loss and optimizer
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)  # Adam optimizer
    loss_fn = nn.BCEWithLogitsLoss()  # "cross entropy loss" for multiple classes, in this case binary cross entropy loss

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
        # train the model
        train_loss, accuracy_train, f1_train = train(model, trainloader, optimizer, loss_fn, device)
        train_loss_tot.append(train_loss)
        accuracies_train.append(accuracy_train)
        f1_scores_train.append(f1_train)
        # evaluate the model on the validation set
        valid_loss, accuracy_val, f1_val, dice = evaluate(model, testloader, loss_fn, device)
        valid_loss_tot.append(valid_loss)
        accuracies_val.append(accuracy_val)
        f1_scores_val.append(f1_val)
        Dice.append(dice)

        # save the model only if the validation loss is improved
        if valid_loss < best_valid_loss:
            data = f"Validation loss improved from {best_valid_loss:2.4f} to {valid_loss:2.4f}. \n Saving checkpoint: {checkpoint_path}"
            print(data)
            best_valid_loss = valid_loss # update the best validation loss

            save_checkpoint(model.state_dict(), checkpoint_path)

    # save the training losses and metrics to a csv file
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
