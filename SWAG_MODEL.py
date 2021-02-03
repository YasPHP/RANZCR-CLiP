# RESNET BASE MODEL

base_model = ResNet50(include_top=False,
                      weights=None,
                      input_shape=(image_size, image_size, 3))

base_model.load_weights("../input/keras-pretrained-models/resnet50_weights_tf_dim_ordering_tf_kernels_notop.h5", by_name=True)

base_model.trainable = False
#-----------------------------------------------------------------------------------------------------

#---------# SWAG DF GENERATORS (TRAIN, VALID, TEST [UNIVERSAL/NON-UNIQUE]) #---------#

## DATFRAME FILTER
SWAG_df_train = df_train[["StudyInstanceUID", "SSwan Ganz Catheter Present"]]
SWAG_df_test = df_test[["StudyInstanceUID", "SSwan Ganz Catheter Present"]]
# print(SWAG_df.head())

# SWAG GENERATORS 
SWAG_dftrain_generator=datagen.flow_from_dataframe(
    dataframe=SWAG_df_train[:21000],
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

SWAG_dfvalid_generator=test_datagen.flow_from_dataframe(
    dataframe=SWAG_df_train[21000:],
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
    dataframe=SWAG_df_test,
    directory=comp_dir+"test",
    x_col="StudyInstanceUID",
    batch_size=1,
    seed=42,
    shuffle=False,
    color_mode="rgb",
    class_mode=None,
    target_size=(image_size,image_size),
    interpolation="bilinear")



#---------# SWAG CATHETER (ABNORMAL, BORDERLINE, NORMAL) #---------#

inp = Input(shape = (image_size,image_size,3))
x = base_model(inp)
x = Flatten()(x)


# OUTPUT FUNNELLING TO SWAG_model
# “binary_crossentropy” as loss function and “sigmoid” as the final layer activation
output11 = Dense(1, activation = 'sigmoid')(x)

# SWAG_MODEL
SWAG_model = Model(inp,[output11])


# STOCHASTIC GRADIENT DESCENT
sgd = SGD(lr=learn_rate, momentum=.9, nesterov=False)

SWAG_model.compile(optimizer=sgd,
              loss = ["binary_crossentropy" for i in range(1)],
              metrics = ["accuracy"])


def generator_wrapper(generator):
    for batch_x,batch_y in generator:
        yield (batch_x,[batch_y[:,i] for i in range(1)])


STEP_SIZE_TRAIN_SWAG = SWAG_dftrain_generator.n//SWAG_dfvalid_generator.batch_size
STEP_SIZE_VALID_SWAG = SWAG_dfvalid_generator.n//SWAG_dfvalid_generator.batch_size
STEP_SIZE_TEST_SWAG = test_generator.n//test_generator.batch_size

SWAG_history = SWAG_model.fit_generator(generator=generator_wrapper(SWAG_dftrain_generator),
                    steps_per_epoch=STEP_SIZE_TRAIN,
                    validation_data=generator_wrapper(SWAG_dfvalid_generator),
                    validation_steps=STEP_SIZE_VALID_SWAG,
                    epochs=num_epochs,verbose=2)

test_generator.reset()
SWAG_pred = SWAG_model.predict_generator(SWAG_dftrain_generator,
                             steps=STEP_SIZE_TEST_SWAG,
                             verbose=1)


# Create Submission df
df_submission = pd.DataFrame(np.squeeze(pred)).transpose()
df_submission.rename(columns=dict(zip([str(i) for i in range(12)], label_cols)))
df_submission["StudyInstanceUID"] = test_files

# SUBMISSION FILE
df_submission.to_csv("submission.csv", index=False)

# GRAPH
epochs = range(1,num_epochs)
plt.plot(SWAG_history.SWAG_history['loss'], label='Training Set')
plt.plot(SWAG_history.SWAG_history['val_loss'], label='Validation Data)')
plt.title('Training and Validation loss')
plt.ylabel('MAE')
plt.xlabel('Num Epochs')
plt.legend(loc="upper left")
plt.show()
plt.savefig("loss.png")
