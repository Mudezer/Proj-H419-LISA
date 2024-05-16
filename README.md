# Proj-H419-LISA

This repository is a computing project as part of the first master year at the polytechnic school of Brussels (EPB).

The chosen project is : Biomedical engineering project in image analysis. [PROJ-H-419] 

The goals of this project are:
- To understand the basic principles of image analysis and processing in the context of machine learning.
- To learn to manipulate some machine learning technology.
- To apply these principles to a concrete problem based on some kaggle competition.
(optional) To search a certain field of application of machine learning in the context of biomedical engineering.
- To do a state of the art on the technology chosen.
For this project, I've mainly chosen to work with PyTorch, a machine learning library for Python mainly focusing on Neural Network.


As Kaggle competition I've chosen :
[Carvana Image Masking Challenge](https://www.kaggle.com/c/carvana-image-masking-challenge/overview)
The objective of this competition is to create a model for binary image segmentation. 
The goal is to predict a mask for each image in the test set. The mask should be 1 where the pixels belong to the object of interest and 0 elsewhere.

The project is divided into 3 main parts:
- Dataset creation
- Model creation and training
- Model testing

## Running the code

### Libraries
You will first need to install the required libraries.

You can do this either by using a python package manager or by running the following command in the terminal:
```bash
pip install -r requirements.txt
```

### Run

To run the code, you can simply run the following command in the terminal:

Generating the datasets:
```bash
python3 src/generate_test_set.py
```

Training the model:
```bash
python3 src/train_unet.py
```

Testing the model:
```bash
python3 src/test_unet.py
```







