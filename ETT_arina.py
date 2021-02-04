base_model = ResNet50(include_top=False,
                      weights=None,
                      input_shape=(image_size, image_size, 3))

base_model.load_weights("../input/keras-pretrained-models/resnet50_weights_tf_dim_ordering_tf_kernels_notop.h5", by_name=True)

base_model.trainable = False

ETT_df_train = df_train[["StudyInstanceUID", "ETT - Abnormal", "ETT - Borderline", "ETT - Normal"]]
ETT_df_test = df_test[["StudyInstanceUID", "ETT - Abnormal", "ETT - Borderline", "ETT - Normal"]]

ETT_dftrain_generator=datagen.flow_from_dataframe(
    dataframe=ETT_train[:21000],
    directory="./miml_dataset/images",
    x_col="StudyInstanceUID",
    y_col=columns,
    batch_size=batch_size,
    seed=42,
    shuffle=True,
    class_mode="raw",
    target_size=(image_size,image_size)

ETT_dfvalid_generator=test_datagen.flow_from_dataframe(
    dataframe=ETT_train[21000:],
    directory="./miml_dataset/images",
    x_col="StudyInstanceUID",
    y_col=columns,
    batch_size=batch_size,
    seed=42,
    shuffle=True,
    class_mode="raw",
    target_size=(image_size,image_size)
    
ETT_test_generator=test_datagen.flow_from_dataframe(
    dataframe=df[21000:],
    directory="./miml_dataset/images",
    x_col="StudyInstanceUID",
    batch_size=1,
    seed=42,
    shuffle=False,
    class_mode=None,
    target_size=(image_size,image_size)
    
inp = Input(shape = (image_size,image_size,3))
x = base_model(inp)
x = Flatten()(x)

output1 = Dense(1, activation = 'sigmoid')(x)
output2 = Dense(1, activation = 'sigmoid')(x)
output3 = Dense(1, activation = 'sigmoid')(x)
output4 = Dense(1, activation = 'sigmoid')(x)
output5 = Dense(1, activation = 'sigmoid')(x)
output6 = Dense(1, activation = 'sigmoid')(x)
output7 = Dense(1, activation = 'sigmoid')(x)
output8 = Dense(1, activation = 'sigmoid')(x)
output9 = Dense(1, activation = 'sigmoid')(x)
output10 = Dense(1, activation = 'sigmoid')(x)

ETT_model = Model(inp,[output1,output2,output3,output4,output5,output6,output7,output8,output9,output10,output11])

ETT_model.compile(optimizers.rmsprop(lr = 0.0001, decay = 1e-6),
    loss = ["binary_crossentropy"]
    metrics = ["accuracy"])
    
def generator_wrapper(generator):
    for batch_x,batch_y in generator:
        yield (batch_x,[batch_y[:,i] for i in range(3)])
        
STEP_SIZE_TRAIN=ETT_train_generator.n//train_generator.batch_size
STEP_SIZE_VALID=valid_generator.n//valid_generator.batch_size
STEP_SIZE_TEST=test_generator.n//test_generator.batch_size
model.fit_generator(generator=generator_wrapper(train_generator),
                    steps_per_epoch=STEP_SIZE_TRAIN)
validation_data=generator_wrapper(valid_generator),
                    validation_steps=STEP_SIZE_VALID,
                    epochs=1,verbose=2
                    
test_generator.reset()
pred=model.predict_generator(test_generator,
      steps=STEP_SIZE_TEST,
      verbose=1)
