# RESNET BASE MODEL

base_model = ResNet50(include_top=False,
                      weights=None,
                      input_shape=(image_size, image_size, 3))

base_model.load_weights("../input/keras-pretrained-models/resnet50_weights_tf_dim_ordering_tf_kernels_notop.h5", by_name=True)

base_model.trainable = False
#-----------------------------------------------------------------------------------------------------

#---------# ETT DF GENERATORS (TRAIN, VALID, TEST [UNIVERSAL/NON-UNIQUE]) #---------#

## DATFRAME FILTER
ETT_df_train = df_train[["StudyInstanceUID", "ETT - Abnormal", "ETT - Borderline", "ETT - Normal"]]
ETT_df_test = df_test[["StudyInstanceUID", "ETT - Abnormal", "ETT - Borderline", "ETT - Normal"]]
# print(ETT_df.head())

# ETT GENERATORS 
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

ETT_dfvalid_generator=test_datagen.flow_from_dataframe(
    dataframe=ETT_df_train[21000:],
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
    dataframe=ETT_df_test,
    directory=comp_dir+"test",
    x_col="StudyInstanceUID",
    batch_size=1,
    seed=42,
    shuffle=False,
    color_mode="rgb",
    class_mode=None,
    target_size=(image_size,image_size),
    interpolation="bilinear")



#---------# ETT CATHETER (ABNORMAL, BORDERLINE, NORMAL) #---------#

inp = Input(shape = (image_size,image_size,3))
x = base_model(inp)
x = Flatten()(x)


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


def generator_wrapper(generator):
    for batch_x,batch_y in generator:
        yield (batch_x,[batch_y[:,i] for i in range(3)])


STEP_SIZE_TRAIN_ETT = ETT_dftrain_generator.n//ETT_dfvalid_generator.batch_size
STEP_SIZE_VALID_ETT = ETT_dfvalid_generator.n//ETT_dfvalid_generator.batch_size
STEP_SIZE_TEST_ETT = test_generator.n//test_generator.batch_size

ETT_history = ETT_model.fit_generator(generator=generator_wrapper(ETT_dftrain_generator),
                    steps_per_epoch=STEP_SIZE_TRAIN,
                    validation_data=generator_wrapper(ETT_dfvalid_generator),
                    validation_steps=STEP_SIZE_VALID_ETT,
                    epochs=num_epochs,verbose=2)

test_generator.reset()
ETT_pred = ETT_model.predict_generator(ETT_dftrain_generator,
                             steps=STEP_SIZE_TEST_ETT,
                             verbose=1)


# Create Submission df
df_submission = pd.DataFrame(np.squeeze(pred)).transpose()
df_submission.rename(columns=dict(zip([str(i) for i in range(12)], label_cols)))
df_submission["StudyInstanceUID"] = test_files

# SUBMISSION FILE
df_submission.to_csv("submission.csv", index=False)

# GRAPH
epochs = range(1,num_epochs)
plt.plot(ETT_history.ETT_history['loss'], label='Training Set')
plt.plot(ETT_history.ETT_history['val_loss'], label='Validation Data)')
plt.title('Training and Validation loss')
plt.ylabel('MAE')
plt.xlabel('Num Epochs')
plt.legend(loc="upper left")
plt.show()
plt.savefig("loss.png")