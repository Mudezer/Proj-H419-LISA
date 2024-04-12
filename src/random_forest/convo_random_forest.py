import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tqdm import tqdm
import joblib

from skimage.io import imread, imshow, imsave
from skimage.transform import resize

from keras.models import Sequential
from keras.layers import Dense, Conv2D, MaxPooling2D, Flatten, Dropout
import os

from sklearn.ensemble import RandomForestClassifier

def get_data(training_folder, mask_folder):
    training_data = []
    mask_data = []

    for img in tqdm(os.listdir(training_folder)):
        path = os.path.join(training_folder, img)
        img = imread(path)
        img = resize(img, (160, 240, 3), anti_aliasing=True)
        training_data.append(img)

    for img in tqdm(os.listdir(mask_folder)):
        path = os.path.join(mask_folder, img)
        img = imread(path).squeeze(0)
        img = resize(img, (160, 240,1), anti_aliasing=True)
        mask_data.append(img)

    training_data = np.array(training_data)
    mask_data = np.array(mask_data)
    # mask_data = np.expand_dims(mask_data, axis=3)
    print(training_data.shape)
    print(mask_data.shape)


    # print()
    return training_data, mask_data

def build_feature_extractor(Height, Width):
    print("Building Feature Extractor")
    activation = 'sigmoid'
    model = Sequential()
    model.add(Conv2D(32, 3, activation = activation, padding = 'same', input_shape = (Height,Width,3)))
    model.add(Conv2D(32,3, activation = activation, padding = 'same', kernel_initializer='he_uniform'))

    return model

def build_classifier():
    model = RandomForestClassifier(n_estimators = 50, random_state = 66)
    return model

def trainRandomForest(train, mask):
    train_ims, train_masks = get_data(train, mask)

    feature_extractor = build_feature_extractor(160, 240)

    print("Extracting Features from training images")

    X = feature_extractor.predict(train_ims)
    X = X.reshape(-1, X.shape[3])

    Y = train_masks.reshape(-1)

    df = pd.DataFrame(data=X)
    df['Label'] = np.uint8(Y)
    df.head()

    df = df[df['Label'] != 0]

    X_RF = df.drop(labels=['Label'], axis=1)
    Y_RF = df['Label']

    RF = RandomForestClassifier(n_estimators = 50, random_state = 66)
    RF.fit(X_RF, Y_RF)

    joblib.dump(RF, "../../output/random_forest/random_forest_model.joblib")


def testRandomForest(test, masks, model):
    # test_ims, test_masks = get_data(test, masks)

    img = imread("../../assets/test/0ce66b539f52_07.jpg")
    img = resize(img, (160, 240, 3), anti_aliasing=True)
    img = np.expand_dims(img, axis=0)

    mask = imread("../../assets/test_masks/0ce66b539f52_07_mask.gif")
    mask = mask.squeeze(0)
    mask = resize(mask, (160, 240, 1), anti_aliasing=True)


    feature_extractor = build_feature_extractor(160, 240)

    print("Extracting Features from test images")

    X = feature_extractor.predict(img)
    X = X.reshape(-1, X.shape[3])

    # Y = test_masks.reshape(-1)
    RF = joblib.load(model)

    shape = (160, 240, 1)

    predictions = RF.predict(X)

    prediction_image = predictions.reshape(mask.shape)

    plt.figure()
    plt.imshow(prediction_image)
    plt.show()

    # return predictions

if __name__ == "__main__":
    train = "../../assets/validation/"
    mask = "../../assets/validation_masks/"

    # trainRandomForest(train, mask)

    test = "../../assets/test/"
    masks = "../../assets/test_masks/"
    model = "../../output/random_forest/random_forest_model.joblib"


    testRandomForest(test, masks, model)

