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
from tensorflow.keras.applications import EfficientNetB4, ResNet50
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
batch_size = 16       #32 changed
num_epochs = 8        #12 changed
learn_rate = 1e-03
df_train.StudyInstanceUID += ".jpg"

# Train-Val = [:21000], [21000:], test_files
# Train-Val-Test (for tuning model) = [:18000], [18000:24000], [24000:]

#----------------EfficientNetB4------------------------------------------------------------------------------

# base_model = ResNet50(include_top=False,
#                       weights=None,
#                       input_shape=(image_size, image_size, 3))
# base_model.load_weights("../input/keras-pretrained-models/resnet50_weights_tf_dim_ordering_tf_kernels_notop.h5", by_name=True)

# for layer in base_model.layers[:-6]:
#     layer.trainable = False

base_model = EfficientNetB4(include_top=False,
                      weights="imagenet",
                      input_shape=(image_size, image_size, 3))
# base_model.load_weights("../input/tfkeras-22-pretrained-and-vanilla-efficientnet/TF2.2_EfficientNetB4_NoTop_ImageNet.h5", by_name=True)
# base_model.trainable = False

for layer in base_model.layers[:-6]:
        if not isinstance(layer, BatchNormalization):
            layer.trainable = False

#----------------IMAGE DATA GENERATOR------------------------------------------------------------------
label_cols=df_train.columns.tolist()
label_cols.remove("StudyInstanceUID")
label_cols.remove("PatientID")
datagen=ImageDataGenerator(rescale=1./255.)
test_datagen=ImageDataGenerator(rescale=1./255.)

#----------------TEST GENERATOR------------------------------------------------------------------------
test_generator=test_datagen.flow_from_dataframe(
    dataframe=df_train[24000:],          # changed from df_test
    directory=comp_dir+"train",          # change this
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


#---------# NGT DF GENERATORS (TRAIN, VALID, TEST [UNIVERSAL/NON-UNIQUE]) #---------#

## DATFRAME FILTER
NGT_df_train = df_train[["StudyInstanceUID", "NGT - Abnormal", "NGT - Borderline", "NGT - Incompletely Imaged", "NGT - Normal"]]

# NGT GENERATORS 
NGT_dftrain_generator=datagen.flow_from_dataframe(
    dataframe=NGT_df_train[:18000],
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
    dataframe=NGT_df_train[18000:24000],
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

inp = Input(shape = (image_size,image_size,3))
x = base_model(inp)
x = Flatten()(x)

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

NGT_history = NGT_model.fit(generator_wrapper(NGT_dftrain_generator, 0, 4),
                    steps_per_epoch=STEP_SIZE_TRAIN_NGT,
                    validation_data=generator_wrapper(NGT_dfvalid_generator, 0, 4),
                    validation_steps=STEP_SIZE_VALID_NGT,
                    epochs=num_epochs,verbose=2)

test_generator.reset()
NGT_pred = NGT_model.predict_generator(test_generator,
                             steps=STEP_SIZE_TEST_NGT,
                             verbose=1)
#-----------------------------------------------------------------------------------------------------


#==========================FINAL SUBMISSION============================#

# Get AUC Score    # TEST THIS

#ETT_model.save     save weights- debug to load file with weights (google)
# y_pred = np.transpose(np.squeeze(pred))

# Get AUC Score
# DOUBLE CHECK
y_pred = np.transpose(np.squeeze(NGT_pred))

y_true = df_train.loc[24000:, label_cols].to_numpy()
aucs = roc_auc_score(y_true, y_pred, average=None)  # predictions per row

print("AUC Scores: ", aucs)
print("Average AUC: ", np.mean(aucs))


# CATHETER COLUMNS
NGT_COLUMNS = ["NGT - Abnormal", "NGT - Borderline", 
         "NGT - Incompletely Imaged", "NGT - Normal"]


# Catheter Dataframes
NGT_df_submission = pd.DataFrame(np.squeeze(NGT_pred).transpose(), columns = NGT_COLUMNS)

# StudyInstanceUID DataFrame
SUID_df_submission = pd.DataFrame({"StudyInstanceUID": test_files})

# concatenated dataframes
catheter_df = [SUID_df_submission, NGT_df_submission]

# FINAL SUBMISSION DF

# puts IDs in first dataframe index
df_submission = pd.concat(catheter_df)

# SUBMISSION FILE
df_submission.to_csv("submission.csv", index=False)


#==========================CATHETER RESULTS GRAPHS============================#

# NGT GRAPH
epochs = range(1,num_epochs)
plt.plot(NGT_history.NGT_history['loss'], label='Training Set')
plt.plot(NGT_history.NGT_history['val_loss'], label='Validation Data)')
plt.title('NGT Training and Validation Loss')
plt.ylabel('MAE')
plt.xlabel('Num Epochs')
plt.legend(loc="upper left")
plt.show()
plt.savefig("NGT_loss.png")
