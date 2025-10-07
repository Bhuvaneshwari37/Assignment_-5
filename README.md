**Multiclass Fish Image Classification with CNN & Pretrained Models**

This repository contains code for classifying fish images into multiple categories using a custom Convolutional Neural Network (CNN) and several pretrained models like VGG16, ResNet50, MobileNet, InceptionV3, and EfficientNetB0. The project includes preprocessing, training, evaluation, deployment, and performance comparison.
Table of Contents
•	Project Overview
•	Installation
•	Imports
•	Dataset
•	Directory Structure
•	Preprocessing
•	Model
•	Training
•	Evaluation
•	Results
•	Usage
•	Class Names
•	Visualizations
•	License
Project Overview
The objective of this project is to classify fish species using deep learning models. It compares the performance of a basic CNN and several pretrained models to identify which architecture performs best for fish image classification.
•	Dataset: Images labeled into 11 fish categories.
•	Techniques: Custom CNN, data augmentation, pretrained model fine-tuning (VGG16, ResNet50, MobileNet, InceptionV3, EfficientNetB0).
•	Evaluation: Metrics include accuracy, precision, recall, F1-score, and confusion matrix.

Installation
!pip install tensorflow matplotlib scikit-learn
Imports
# File Handling and Image Processing
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.optimizers import Adam
# Data Handling
import pandas as pd
import numpy as np

# Visualization
import seaborn as sns

# Metrics
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
Dataset
The dataset contains labeled fish images categorized into:
•	train/: Training data
•	val/: Validation data
•	test/: Testing data
Directory Structure
Dataset/
├── train/
├── val/
├── test/
Preprocessing
•	Resize images to 1./255
•	Random horizontal flips
•	Rotation between 20
•	Random affine transforms
•	Normalize using ImageNet mean and std
Model
CNN Model
•	2 convolutional layers + ReLU + MaxPooling + Softmax
•	Dropout layers
Pretrained Models
•	VGG16, ResNet50, MobileNet, InceptionV3, EfficientNetB0
•	Custom final layers adapted to 11 classes
Loss and Optimizer
•	Loss: CrossEntropyLoss
•	Optimizer: Adam (lr=0.001)
Training
•	Trained for 5  epochs
•	Batch size: 195
•	Used train_loader and val_loader for training and validation
•	DataLoader with shuffle=True for training
Each pretrained model was trained using:
•	train_loader for training
•	val_loader for validation
•	Batch size = 195
•	Evaluation done on eval_loader or test_loader
•	Metrics like accuracy, precision, recall, and F1 were computed per model
Evaluation
Metrics used:
•	Accuracy
•	Precision
•	Recall
•	F1-Score
•	Confusion Matrix
Results
Model Performance Comparison
vgg16           : 0.9679
resnet50        : 0.9908
mobilenet       : 0.9899
inceptionv3     : 0.9679
efficientnetb0  : 0.9881

Usage
Run the Streamlit app from Colab or local:
http://localhost:8501
Class Names
The 11 fish species classes used in the model are:
•	animal fish
•	animal fish bass
•	fish sea_food black_sea_sprat
•	fish sea_food gilt_head_bream
•	fish sea_food hourse_mackerel
•	fish sea_food red_mullet
•	fish sea_food red_sea_bream
•	fish sea_food sea_bass
•	fish sea_food shrimp
•	fish sea_food striped_red_mullet
•	fish sea_food trout

