#-----------IMPORTS-----------------------------------------------------------------------------
import pandas as pd 
import numpy as np
import tensorflow as tf
import os
import matplotlib.pyplot as plt
import seaborn as sns
import PIL
from typing import List

# EfficientNet
from tensorflow.keras.applications import EfficientNetB7, ResNet50
from tensorflow.keras.applications.efficientnet import preprocess_input

# Data Augmentation
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Model Layers
from tensorflow.keras import Model, Input
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

# Get Training/Testing Data Paths
test_files = os.listdir(comp_dir+"test")      

df_test = pd.DataFrame({"StudyInstanceUID": test_files})

image_size = 512
batch_size = 16
num_epochs = 12
learn_rate = 1e-03
df_train.StudyInstanceUID += ".jpg"

# Train-Val = [:21000], [21000:], test_files
# Train-Val-Test (for tuning model) = [:18000], [18000:24000], [24000:]

#----------------RESNET------------------------------------------------------------------------------
base_model = ResNet50(include_top=False,
                      weights=None,
                      input_shape=(image_size, image_size, 3))

base_model.load_weights("../input/keras-pretrained-models/resnet50_weights_tf_dim_ordering_tf_kernels_notop.h5", by_name=True)

base_model.trainable = False

#----------------IMAGE DATA GENERATOR------------------------------------------------------------------
label_cols=df_train.columns.tolist()
label_cols.remove("StudyInstanceUID")
label_cols.remove("PatientID")
datagen=ImageDataGenerator(rescale=1./255.)
test_datagen=ImageDataGenerator(rescale=1./255.)

#----------------TEST GENERATOR------------------------------------------------------------------------
test_generator=test_datagen.flow_from_dataframe(
    dataframe=df_test,
    directory=comp_dir+"test",          # Change this
    x_col="StudyInstanceUID",
    batch_size=1,
    seed=42,
    shuffle=False,
    color_mode="rgb",
    class_mode=None,
    target_size=(image_size,image_size),
    interpolation="bilinear")
#-----------------------------------------------------------------------------------------------------

#----------------GENERATOR WRAPPER--------------------------------------------------------------------

def generator_wrapper(generator, start, end):
    for batch_x,batch_y in generator:
        yield (batch_x,[batch_y[:,i] for i in range(start, end)])
#-----------------------------------------------------------------------------------------------------

#---------# CVC DF GENERATORS (TRAIN, VALID, TEST [UNIVERSAL/NON-UNIQUE]) #---------#

## DATFRAME FILTER
CVC_df_train = df_train[["StudyInstanceUID", "CVC - Abnormal", "CVC - Borderline", "CVC - Normal"]]

# CVC GENERATORS 
CVC_dftrain_generator=datagen.flow_from_dataframe(
    dataframe=CVC_df_train[:21000],
    directory=comp_dir+"train",
    x_col="StudyInstanceUID",
    y_col=["CVC - Abnormal", "CVC - Borderline", "CVC - Normal"],
    batch_size=batch_size,
    seed=42,
    shuffle=True,
    color_mode="rgb",
    class_mode="raw",
    target_size=(image_size,image_size),
    interpolation="bilinear")

CVC_dfvalid_generator=test_datagen.flow_from_dataframe(
    dataframe=CVC_df_train[21000:],
    directory=comp_dir+"train",
    x_col="StudyInstanceUID",
    y_col=["CVC - Abnormal", "CVC - Borderline", "CVC - Normal"],
    batch_size=batch_size,
    seed=42,
    shuffle=True,
    color_mode="rgb",
    class_mode="raw",
    target_size=(image_size,image_size),
    interpolation="bilinear")
    
#---------# CVC CATHETER (ABNORMAL, BORDERLINE, NORMAL) #---------#

inp = Input(shape = (image_size,image_size,3))
x = base_model(inp)
x = Flatten()(x)

# OUTPUT FUNNELLING TO CVC_model
# “binary_crossentropy” as loss function and “sigmoid” as the final layer activation
output4 = Dense(1, activation = 'sigmoid')(x)
output5 = Dense(1, activation = 'sigmoid')(x)
output6 = Dense(1, activation = 'sigmoid')(x)

# CVC_MODEL
CVC_model = Model(inp,[output4,output5,output6])


# STOCHASTIC GRADIENT DESCENT
sgd = SGD(lr=learn_rate, momentum=.9, nesterov=False)

CVC_model.compile(optimizer=sgd,
              loss = ["binary_crossentropy" for i in range(3)],
              metrics = ["accuracy"])


STEP_SIZE_TRAIN_CVC = CVC_dftrain_generator.n//CVC_dfvalid_generator.batch_size
STEP_SIZE_VALID_CVC = CVC_dfvalid_generator.n//CVC_dfvalid_generator.batch_size
STEP_SIZE_TEST_CVC = test_generator.n//test_generator.batch_size

CVC_history = CVC_model.fit_generator(generator=generator_wrapper(CVC_dftrain_generator, 7, 10),
                    steps_per_epoch=STEP_SIZE_TRAIN_CVC,
                    validation_data=generator_wrapper(CVC_dfvalid_generator, 7, 10),
                    validation_steps=STEP_SIZE_VALID_CVC,
                    epochs=num_epochs,verbose=2)

test_generator.reset()
CVC_pred = CVC_model.predict_generator(CVC_dftrain_generator,
                             steps=STEP_SIZE_TEST_CVC,
                             verbose=1)
#-----------------------------------------------------------------------------------------------------

#---------# ETT DF GENERATORS (TRAIN, VALID, TEST [UNIVERSAL/NON-UNIQUE]) #---------#

## DATFRAME FILTER
ETT_df_train = df_train[["StudyInstanceUID", "ETT - Abnormal", "ETT - Borderline", "ETT - Normal"]]

# ETT GENERATORS 
ETT_dftrain_generator=datagen.flow_from_dataframe(
    dataframe=ETT_df_train[:21000],
    directory=comp_dir+"train",
    x_col="StudyInstanceUID",
    y_col=["ETT - Abnormal", "ETT - Borderline", "ETT - Normal"],
    batch_size=batch_size,
    seed=42,
    shuffle=True,
    color_mode="rgb",
    class_mode="raw",
    target_size=(image_size,image_size),
    interpolation="bilinear")

ETT_dfvalid_generator=test_datagen.flow_from_dataframe(
    dataframe=ETT_df_train[21000:],
    directory=comp_dir+"train",
    x_col="StudyInstanceUID",
    y_col=["ETT - Abnormal", "ETT - Borderline", "ETT - Normal"],
    batch_size=batch_size,
    seed=42,
    shuffle=True,
    color_mode="rgb",
    class_mode="raw",
    target_size=(image_size,image_size),
    interpolation="bilinear")
#---------# ETT CATHETER (ABNORMAL, BORDERLINE, NORMAL) #---------#

# OUTPUT FUNNELLING TO ETT_model
# “binary_crossentropy” as loss function and “sigmoid” as the final layer activation
output1 = Dense(1, activation = 'sigmoid')(x)
output2 = Dense(1, activation = 'sigmoid')(x)
output3 = Dense(1, activation = 'sigmoid')(x)

# ETT_MODEL
ETT_model = Model(inp,[output1,output2,output3])


# STOCHASTIC GRADIENT DESCENT
sgd = SGD(lr=learn_rate, momentum=.9, nesterov=False)

ETT_model.compile(optimizer=sgd,
              loss = ["binary_crossentropy" for i in range(3)],
              metrics = ["accuracy"])


STEP_SIZE_TRAIN_ETT = ETT_dftrain_generator.n//ETT_dfvalid_generator.batch_size
STEP_SIZE_VALID_ETT = ETT_dfvalid_generator.n//ETT_dfvalid_generator.batch_size
STEP_SIZE_TEST_ETT = test_generator.n//test_generator.batch_size

ETT_history = ETT_model.fit_generator(generator=generator_wrapper(ETT_dftrain_generator, 0, 3),
                    steps_per_epoch=STEP_SIZE_TRAIN_ETT,
                    validation_data=generator_wrapper(ETT_dfvalid_generator, 0, 3),
                    validation_steps=STEP_SIZE_VALID_ETT,
                    epochs=num_epochs,verbose=2)

test_generator.reset()
ETT_pred = ETT_model.predict_generator(ETT_dftrain_generator,
                             steps=STEP_SIZE_TEST_ETT,
                             verbose=1)


# Get AUC Score    # TEST THIS
# y_pred = np.transpose(np.squeeze(pred))
# y_true = df_train.loc[24000:, label_cols].to_numpy()
# aucs = roc_auc_score(y_true, y_pred, average=None)

# print("AUC Scores: ", aucs)
# print("Average AUC: ", np.mean(aucs))

#-----------------------------------------------------------------------------------------------------

#---------# NGT DF GENERATORS (TRAIN, VALID, TEST [UNIVERSAL/NON-UNIQUE]) #---------#

## DATFRAME FILTER
NGT_df_train = df_train[["StudyInstanceUID", "NGT - Abnormal", "NGT - Borderline", "NGT - Incompletely Imaged", "NGT - Normal"]]

# NGT GENERATORS 
NGT_dftrain_generator=datagen.flow_from_dataframe(
    dataframe=NGT_df_train[:21000],
    directory=comp_dir+"train",
    x_col="StudyInstanceUID",
    y_col=["NGT - Abnormal", "NGT - Borderline", "NGT - Incompletely Imaged", "NGT - Normal"],
    batch_size=batch_size,
    seed=42,
    shuffle=True,
    color_mode="rgb",
    class_mode="raw",
    target_size=(image_size,image_size),
    interpolation="bilinear")

NGT_dfvalid_generator=test_datagen.flow_from_dataframe(
    dataframe=NGT_df_train[21000:],
    directory=comp_dir+"train",
    x_col="StudyInstanceUID",
    y_col=["NGT - Abnormal", "NGT - Borderline", "NGT - Incompletely Imaged", "NGT - Normal"],
    batch_size=batch_size,
    seed=42,
    shuffle=True,
    color_mode="rgb",
    class_mode="raw",
    target_size=(image_size,image_size),
    interpolation="bilinear")

#---------# NGT CATHETER (ABNORMAL, BORDERLINE, NORMAL) #---------#
# OUTPUT FUNNELLING TO NGT_model
# “binary_crossentropy” as loss function and “sigmoid” as the final layer activation
output7 = Dense(1, activation = 'sigmoid')(x)
output8 = Dense(1, activation = 'sigmoid')(x)
output9 = Dense(1, activation = 'sigmoid')(x)
output10 = Dense(1, activation = 'sigmoid')(x)

# NGT_MODEL
NGT_model = Model(inp,[output7,output8,output9,output10])

# STOCHASTIC GRADIENT DESCENT
sgd = SGD(lr=learn_rate, momentum=.9, nesterov=False)

NGT_model.compile(optimizer=sgd,
              loss = ["binary_crossentropy" for i in range(4)],
              metrics = ["accuracy"])


STEP_SIZE_TRAIN_NGT = NGT_dftrain_generator.n//NGT_dfvalid_generator.batch_size
STEP_SIZE_VALID_NGT = NGT_dfvalid_generator.n//NGT_dfvalid_generator.batch_size
STEP_SIZE_TEST_NGT = test_generator.n//test_generator.batch_size

NGT_history = NGT_model.fit_generator(generator=generator_wrapper(NGT_dftrain_generator, 3, 7),
                    steps_per_epoch=STEP_SIZE_TRAIN_NGT,
                    validation_data=generator_wrapper(NGT_dfvalid_generator, 3, 7),
                    validation_steps=STEP_SIZE_VALID_NGT,
                    epochs=num_epochs,verbose=2)

test_generator.reset()
NGT_pred = NGT_model.predict_generator(NGT_dftrain_generator,
                             steps=STEP_SIZE_TEST_NGT,
                             verbose=1)
#-----------------------------------------------------------------------------------------------------

#---------# SWAG DF GENERATORS (TRAIN, VALID, TEST [UNIVERSAL/NON-UNIQUE]) #---------#

## DATFRAME FILTER
SWAG_df_train = df_train[["StudyInstanceUID", "Swan Ganz Catheter Present"]]

# SWAG GENERATORS 
SWAG_dftrain_generator=datagen.flow_from_dataframe(
    dataframe=SWAG_df_train[:21000],
    directory=comp_dir+"train",
    x_col="StudyInstanceUID",
    y_col=["Swan Ganz Catheter Present"],
    batch_size=batch_size,
    seed=42,
    shuffle=True,
    color_mode="rgb",
    class_mode="raw",
    target_size=(image_size,image_size),
    interpolation="bilinear")

SWAG_dfvalid_generator=test_datagen.flow_from_dataframe(
    dataframe=SWAG_df_train[21000:],
    directory=comp_dir+"train",
    x_col="StudyInstanceUID",
    y_col=["Swan Ganz Catheter Present"],
    batch_size=batch_size,
    seed=42,
    shuffle=True,
    color_mode="rgb",
    class_mode="raw",
    target_size=(image_size,image_size),
    interpolation="bilinear")
#---------# SWAG CATHETER (ABNORMAL, BORDERLINE, NORMAL) #---------#

# OUTPUT FUNNELLING TO SWAG_model
# “binary_crossentropy” as loss function and “sigmoid” as the final layer activation
output11 = Dense(1, activation = 'sigmoid')(x)

# SWAG_MODEL
SWAG_model = Model(inp,[output11])


# STOCHASTIC GRADIENT DESCENT
sgd = SGD(lr=learn_rate, momentum=.9, nesterov=False)

SWAG_model.compile(optimizer=sgd,
              loss = "binary_crossentropy",
              metrics = ["accuracy"])
              

STEP_SIZE_TRAIN_SWAG = SWAG_dftrain_generator.n//SWAG_dfvalid_generator.batch_size
STEP_SIZE_VALID_SWAG = SWAG_dfvalid_generator.n//SWAG_dfvalid_generator.batch_size
STEP_SIZE_TEST_SWAG = test_generator.n//test_generator.batch_size

SWAG_history = SWAG_model.fit_generator(generator=generator_wrapper(SWAG_dftrain_generator, 10, 11),
                    steps_per_epoch=STEP_SIZE_TRAIN_SWAG,
                    validation_data=generator_wrapper(SWAG_dfvalid_generator, 10, 11),
                    validation_steps=STEP_SIZE_VALID_SWAG,
                    epochs=num_epochs,verbose=2)

test_generator.reset()
SWAG_pred = SWAG_model.predict_generator(SWAG_dftrain_generator,
                             steps=STEP_SIZE_TEST_SWAG,
                             verbose=1)

#==========================FINAL SUBMISSION============================#

# CATHETER COLUMNS
ETT_COLUMNS = ["ETT - Abnormal", "ETT - Borderline", "ETT - Normal"]
NGT_COLUMNS = ["NGT - Abnormal", "NGT - Borderline", 
         "NGT - Incompletely Imaged", "NGT - Normal"]
CVC_COLUMNS = ["CVC - Abnormal", "CVC - Borderline", 
         "CVC - Normal"]
SWAG_COLUMN = ["Swan Ganz Catheter Present"]

# Catheter Dataframes
ETT_df_submission = pd.DataFrame(np.squeeze(ETT_pred).transpose(), columns = ETT_COLUMNS)
NGT_df_submission = pd.DataFrame(np.squeeze(NGT_pred).transpose(), columns = NGT_COLUMNS)
CVC_df_submission = pd.DataFrame(np.squeeze(CVC_pred).transpose(), columns = CVC_COLUMNS)
SWAG_df_submission = pd.DataFrame(np.squeeze(SWAG_pred).transpose(), columns = SWAG_COLUMN)

# StudyInstanceUID DataFrame
SUID_df_submission = pd.DataFrame({"StudyInstanceUID": test_files})

# concatenated dataframes
catheter_df = [SUID_df_submission, ETT_df_submission, NGT_df_submission, CVC_df_submission, SWAG_df_submission]

# FINAL SUBMISSION DF

# puts IDs in first dataframe index
df_submission = pd.concat(catheter_df)

# SUBMISSION FILE
df_submission.to_csv("submission.csv", index=False)



# # GRAPH (STILL NEED TO FIX)
# epochs = range(1,num_epochs)
# plt.plot(SWAG_history.SWAG_history['loss'], label='Training Set')
# plt.plot(SWAG_history.SWAG_history['val_loss'], label='Validation Data)')
# plt.title('Training and Validation loss')
# plt.ylabel('MAE')
# plt.xlabel('Num Epochs')
# plt.legend(loc="upper left")
# plt.show()
# plt.savefig("loss.png")
