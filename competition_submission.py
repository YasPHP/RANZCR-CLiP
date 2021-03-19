import pandas as pd 
import numpy as np
import tensorflow as tf
import os

# EfficientNet
from tensorflow.keras.applications import EfficientNetB4
from tensorflow.keras.applications.efficientnet import preprocess_input

# Data Augmentation
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Model Layers
from tensorflow.keras import Model, Input
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, Conv2D, MaxPool2D , Flatten, Dropout, BatchNormalization

# Compiling and Callbacks
from tensorflow.keras.optimizers import SGD,Adam
from tensorflow.keras.losses import BinaryCrossentropy
from tensorflow.keras.callbacks import ReduceLROnPlateau, ModelCheckpoint, EarlyStopping

# Sklearn
from sklearn.metrics import roc_auc_score
#-----------------------------------------------------------------------------------------------------
# Competition Directory
comp_dir="/kaggle/input/ranzcr-clip-catheter-line-classification/"

# Get Training Data Labels
df_train=pd.read_csv(comp_dir+"train.csv").sample(frac=1).reset_index(drop=True)

# Get Testing Data Paths
test_files = os.listdir(comp_dir+"test")
df_test = pd.DataFrame({"StudyInstanceUID": test_files})

# Parameters
image_size = 380

# Get Labels
label_cols=df_train.columns.tolist()
label_cols.remove("StudyInstanceUID")
label_cols.remove("PatientID")


# Get Test Dataset Generator
test_datagen=ImageDataGenerator()

test_generator=test_datagen.flow_from_dataframe(
    dataframe=df_test,
    directory=comp_dir+"test",    # Change this
    x_col="StudyInstanceUID",
    batch_size=1,
    seed=42,
    shuffle=False,
    color_mode="rgb",
    class_mode=None,
    target_size=(image_size,image_size),
    interpolation="bilinear")

STEP_SIZE_TEST = test_generator.n//test_generator.batch_size

# Load model from H5 Model
model = load_model("../input/ranzcr-clip-big-models/big_model.h5")

# Predict
pred = model.predict(test_generator,
                     steps=STEP_SIZE_TEST,
                     verbose=1)

# Create Submission df
df_submission = pd.DataFrame()
df_submission["StudyInstanceUID"] = test_files
df_submission["StudyInstanceUID"] = df_submission["StudyInstanceUID"].map(lambda x: x.replace(".jpg",""))
df_preds = pd.DataFrame(np.squeeze(pred)).transpose()
df_preds = df_preds.rename(columns=dict(zip([i for i in range(11)], label_cols)))
df_submission = pd.concat([df_submission, df_preds], axis=1)

# Save Submission
df_submission.to_csv("submission.csv", index=False)
