# -*- coding: utf-8 -*-
"""AMY Model.ipynb
Automatically generated by Colaboratory.
Original file is located at
    https://colab.research.google.com/drive/1iFLUhEe1xMrKgRqRMjRQH2IHVPXIebNg
"""

#import numpy as np 
import pandas as pd 
import tensorflow as tf
import os
import matplotlib.pyplot as plt
import seaborn as sns
import PIL
from typing import List


import tensorflow as tf

# EfficientNet
from tensorflow.keras.applications import EfficientNetB7, ResNet50
from tensorflow.keras.applications.efficientnet import preprocess_input

# Data Augmentation
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Model Layers
from tensorflow.keras import Model, Input
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, MaxPool2D , Flatten, Dropout, BatchNormalization

# Compiling and Callbacks
from tensorflow.keras.optimizers import SGD,Adam
from tensorflow.keras.losses import BinaryCrossentropy
from tensorflow.keras.callbacks import ReduceLROnPlateau, ModelCheckpoint, EarlyStopping
#-----------------------------------------------------------------------------------------------------
# Competition Directory
comp_dir="/kaggle/input/ranzcr-clip-catheter-line-classification/"

# Get Training Data Labels
df_train=pd.read_csv(comp_dir+"train.csv").sample(frac=1).reset_index(drop=True)

# Get Training/Testing Data Paths
test_files = os.listdir(comp_dir+"test")      

df_test = pd.DataFrame({"StudyInstanceUID": test_files})

image_size = 256
batch_size = 16
num_epochs = 15
learn_rate = 1e-03
df_train.StudyInstanceUID += ".jpg"

# -----------------------------------------------------------------------------------------------------
#PRETRAINED RESNET CNN
base_model = ResNet50(include_top=False, 
                                    weights="imagenet", 
                                    input_shape=(image_size, image_size, 3))
#-----------------------------------------------------------------------------------------------------
label_cols=df_train.columns.tolist()
label_cols.remove("StudyInstanceUID")
label_cols.remove("PatientID")
datagen=ImageDataGenerator(rescale=1./255.)
test_datagen=ImageDataGenerator(rescale=1./255.)
# -----------------------------------------------------------------------------------------------------
##TEST AND VALIDATION GENERATORS
#Only need one at the beginning right? -> REMEMBER to bring up in meeting!
valid_generator=test_datagen.flow_from_dataframe(
    dataframe=df_train[21000:],
    directory=comp_dir+"train",
    x_col="StudyInstanceUID",
    y_col=label_cols,
    batch_size=batch_size,
    seed=42,
    shuffle=True,
    color_mode="rgb",
    class_mode="raw",
    target_size=(image_size,image_size),
    interpolation="bilinear")

test_generator=test_datagen.flow_from_dataframe(
    dataframe=df_test,
    directory=comp_dir+"test",
    x_col="StudyInstanceUID",
    batch_size=1,
    seed=42,
    shuffle=False,
    color_mode="rgb",
    class_mode=None,
    target_size=(image_size,image_size),
    interpolation="bilinear")
# -----------------------------------------------------------------------------------------------------

# ETT CATHETER (ABNORMAL, BORDERLINE, NORMAL)
# DATFRAME FILTER
ETT_df_train = df_train[["StudyInstanceUID", "ETT - Abnormal", "ETT - Borderline", "ETT - Normal"]]

# ETT TRAIN GENERATOR
ETT_dftrain_generator=datagen.flow_from_dataframe(
    dataframe=ETT_df_train[:21000],
    directory=comp_dir+"train",
    x_col="StudyInstanceUID",
    y_col=label_cols,
    batch_size=batch_size,
    seed=42,
    shuffle=True,
    color_mode="rgb",
    class_mode="raw",
    target_size=(image_size,image_size),
    interpolation="bilinear")

## CONSISTENT PARAMETERS                                      
inp = Input(shape = (image_size,image_size,3))
x = base_model(inp)
x = Flatten()(x)    

## BUILDING THE MODEL (Keras’s Function API )                                      
# Loss Function -> Binary Cross Entropy
# Final Activation Layer -> Sigmoid 
output1 = Dense(1, activation = 'sigmoid')(x)
output2 = Dense(1, activation = 'sigmoid')(x)
output3 = Dense(1, activation = 'sigmoid')(x)


ett_model = Model(inp,[output1,output2,output3])
ett_model.compile(optimizers.rmsprop(lr = 0.0001, decay = 1e-6),
loss = ["binary_crossentropy","binary_crossentropy","binary_crossentropy"],metrics = ["accuracy"])             
                                           

## GENERATOR WRAPPER (3 outputs)
def generator_wrapper(generator):
    for batch_x,batch_y in generator:
        yield (batch_x,[batch_y[:,i] for i in range(3)])

                                           
## FITTING AND TRAINING THE MODEL                                        
ETT_STEP_SIZE_TRAIN=ETT_dftrain_generator.n//ETT_dftrain_generator.batch_size
ETT_STEP_SIZE_VALID=valid_generator.n//valid_generator.batch_size
ETT_STEP_SIZE_TEST=test_generator.n//test_generator.batch_size
ett_model.fit_generator(generator=train_generator,
                    steps_per_epoch=ETT_STEP_SIZE_TRAIN,
                    validation_data=valid_generator,
                    validation_steps=ETT_STEP_SIZE_VALID,
                    epochs=10
)
                                           
                                           
## PREDICITING THE OUTPUT
#Confused about the test/valid_generator -> REMEMBER to bring up in meeting                                                    
test_generator.reset()
pred=ett_model.predict_generator(test_generator,
steps=ETT_STEP_SIZE_TEST,verbose=1)
# -----------------------------------------------------------------------------------------------------
# CVC CATHETER (ABNORMAL, BORDERLINE, NORMAL)
## DATFRAME FILTER
CVC_df_train = df_train[["StudyInstanceUID", "CVC - Abnormal", "CVC - Borderline", "CVC - Normal"]]

# CVC TRAIN GENERATOR
CVC_dftrain_generator=datagen.flow_from_dataframe(
    dataframe=CVC_df_train[:21000],
    directory=comp_dir+"train",
    x_col="StudyInstanceUID",
    y_col=label_cols,
    batch_size=batch_size,
    seed=42,
    shuffle=True,
    color_mode="rgb",
    class_mode="raw",
    target_size=(image_size,image_size),
    interpolation="bilinear")

## CONSISTENT PARAMETERS                                      
inp = Input(shape = (image_size,image_size,3))
x = base_model(inp)
x = Flatten()(x)    
          
# Loss Function -> Binary Cross Entropy
# Final Activation Layer -> Sigmoid 
output4 = Dense(1, activation = 'sigmoid')(x)
output5 = Dense(1, activation = 'sigmoid')(x)
output6 = Dense(1, activation = 'sigmoid')(x)
cvc_model = Model(inp,[output4,output5,output6])
cvc_model.compile(optimizers.rmsprop(lr = 0.0001, decay = 1e-6),
loss = ["binary_crossentropy","binary_crossentropy","binary_crossentropy"],metrics = ["accuracy"])             
                                           

## GENERATOR WRAPPER (3 outputs)
def generator_wrapper(generator):
    for batch_x,batch_y in generator:
        yield (batch_x,[batch_y[:,i] for i in range(3)])

                                           
## FITTING AND TRAINING THE MODEL                                        
CVC_STEP_SIZE_TRAIN=CVC_dftrain_generator.n//CVC_dftrain_generator.batch_size
CVC_STEP_SIZE_VALID=valid_generator.n//valid_generator.batch_size
CVC_STEP_SIZE_TEST=test_generator.n//test_generator.batch_size
cvc_model.fit_generator(generator=train_generator,
                    steps_per_epoch=CVC_STEP_SIZE_TRAIN,
                    validation_data=valid_generator,
                    validation_steps=CVC_STEP_SIZE_VALID,
                    epochs=10
)
                                           
                                           
## PREDICITING THE OUTPUT
                                                                                 
test_generator.reset()
pred=model.predict_generator(test_generator,
steps=CVC_STEP_SIZE_TEST,verbose=1)
# -----------------------------------------------------------------------------------------------------
# NGT CATHETER (ABNORMAL, BORDERLINE, NORMAL)
## DATFRAME FILTER
NGT_df_train = df_train[["StudyInstanceUID", "NGT - Abnormal", "NGT - Borderline", "NGT - Incompletely Imaged", "NGT - Normal"]]

# NGT GENERATORS 
NGT_dftrain_generator=datagen.flow_from_dataframe(
    dataframe=NGT_df_train[:21000],
    directory=comp_dir+"train",
    x_col="StudyInstanceUID",
    y_col=label_cols,
    batch_size=batch_size,
    seed=42,
    shuffle=True,
    color_mode="rgb",
    class_mode="raw",
    target_size=(image_size,image_size),
    interpolation="bilinear")

## CONSISTENT PARAMETERS                                      
inp = Input(shape = (image_size,image_size,3))
x = base_model(inp)
x = Flatten()(x)                

# Loss Function -> Binary Cross Entropy
# Final Activation Layer -> Sigmoid 
output7 = Dense(1, activation = 'sigmoid')(x)
output8 = Dense(1, activation = 'sigmoid')(x)
output9 = Dense(1, activation = 'sigmoid')(x)
output10 = Dense(1, activation = 'sigmoid')(x)
ngt_model = Model(inp,[output7,output8,output9,output10])
ngt_model.compile(optimizers.rmsprop(lr = 0.0001, decay = 1e-6),
loss = ["binary_crossentropy","binary_crossentropy","binary_crossentropy","binary_crossentropy"],metrics = ["accuracy"])             
                                           

## GENERATOR WRAPPER (4 outputs)
def generator_wrapper(generator):
    for batch_x,batch_y in generator:
        yield (batch_x,[batch_y[:,i] for i in range(3)])

                                           
## FITTING AND TRAINING THE MODEL                                        
NGT_STEP_SIZE_TRAIN=NGT_dftrain_generator.n//NGT_dftrain_generator.batch_size
NGT_STEP_SIZE_VALID=valid_generator.n//valid_generator.batch_size
NGT_STEP_SIZE_TEST=test_generator.n//test_generator.batch_size
ngt_model.fit_generator(generator=NGT_dftrain_generator,
                    steps_per_epoch=NGT_STEP_SIZE_TRAIN,
                    validation_data=valid_generator,
                    validation_steps=NGT_STEP_SIZE_VALID,
                    epochs=10
)
                                           
                                           
## PREDICITING THE OUTPUT                                                   
test_generator.reset()
pred=ngt_model.predict_generator(test_generator,
steps=NGT_STEP_SIZE_TEST,verbose=1)
# -----------------------------------------------------------------------------------------------------
#SWAN GANZ CATHETER (Present)  
#DATAFRAME FILTER     
SG_df_train = df_train[["StudyInstanceUID", "Swan Ganz Catheter Present"]]

# ETT TRAIN GENERATOR
SG_dftrain_generator=datagen.flow_from_dataframe(
    dataframe=SG_df_train[:21000],
    directory=comp_dir+"train",
    x_col="StudyInstanceUID",
    y_col=label_cols,
    batch_size=batch_size,
    seed=42,
    shuffle=True,
    color_mode="rgb",
    class_mode="raw",
    target_size=(image_size,image_size),
    interpolation="bilinear")

## CONSISTENT PARAMETERS                                      
inp = Input(shape = (image_size,image_size,3))
x = base_model(inp)
x = Flatten()(x)    

## BUILDING THE MODEL (Keras’s Function API )                                      
# “binary_crossentropy” as loss function and “sigmoid” as the final layer activation
output11 = Dense(1, activation = 'sigmoid')(x)
sg_model = Model(inp,[output11])
sg_model.compile(optimizers.rmsprop(lr = 0.0001, decay = 1e-6),
loss = ["binary_crossentropy"],metrics = ["accuracy"])             
                                           
## GENERATOR WRAPPER (3 outputs)
def generator_wrapper(generator):
    for batch_x,batch_y in generator:
        yield (batch_x,[batch_y[:,i] for i in range(1)])

                                           
## FITTING AND TRAINING THE MODEL                                        
SG_STEP_SIZE_TRAIN=ETT_dftrain_generator.n//ETT_dftrain_generator.batch_size
SG_STEP_SIZE_VALID=valid_generator.n//valid_generator.batch_size
SG_STEP_SIZE_TEST=test_generator.n//test_generator.batch_size
sg_model.fit_generator(generator=train_generator,
                    steps_per_epoch=SG_STEP_SIZE_TRAIN,
                    validation_data=valid_generator,
                    validation_steps=SG_STEP_SIZE_VALID,
                    epochs=10
)
                                           
                                           
## PREDICITING THE OUTPUT

#Confused about the test/valid_generator -> REMEMBER to bring up in meeting                                                    
test_generator.reset()
pred=sg_model.predict_generator(test_generator,
steps=SG_STEP_SIZE_TEST,verbose=1)                            
 

# ----------------------------------------------------------------------------------------------------- 
#FINAL OUTPUT -> putting together all four predictions      
##Confused on how to put all the predictions together->REMEMBER to bring up in meeting                                      
predictions = pred_bool.astype(int)
columns=["ETT - Abnormal", "ETT - Borderline", "ETT - Normal","NGT - Abnormal", "NGT - Borderline", 
         "NGT - Incompletely Imaged", "NGT - Normal", "CVC - Abnormal", "CVC - Borderline", 
         "CVC - Normal", "Swan Ganz Catheter Present"]

#columns should be the same order of y_col
results=pd.DataFrame(predictions, columns=columns)
results["StudyInstanceUID"]=test_generator.StudyInstanceUID
ordered_cols=["StudyInstanceUID"]+columns
results=results[ordered_cols]#To get the same column order
results.to_csv("results.csv",index=False)
