# RESNET BASE MODEL

base_model = ResNet50(include_top=False,
                      weights=None,
                      input_shape=(image_size, image_size, 3))

base_model.load_weights("../input/keras-pretrained-models/resnet50_weights_tf_dim_ordering_tf_kernels_notop.h5", by_name=True)

base_model.trainable = False
#-----------------------------------------------------------------------------------------------------

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

model.compile(optimizer=sgd,
              loss = ["binary_crossentropy" for i in range(3)],
              metrics = ["accuracy"])


def generator_wrapper(generator):
    for batch_x,batch_y in generator:
        yield (batch_x,[batch_y[:,i] for i in range(3)])


STEP_SIZE_TRAIN = train_generator.n//train_generator.batch_size
STEP_SIZE_VALID = valid_generator.n//valid_generator.batch_size
STEP_SIZE_TEST = test_generator.n//test_generator.batch_size

history = model.fit_generator(generator=generator_wrapper(train_generator),
                    steps_per_epoch=STEP_SIZE_TRAIN,
                    validation_data=generator_wrapper(valid_generator),
                    validation_steps=STEP_SIZE_VALID,
                    epochs=num_epochs,verbose=2)

test_generator.reset()
pred = model.predict_generator(test_generator,
                             steps=STEP_SIZE_TEST,
                             verbose=1)


# Create Submission df
df_submission = pd.DataFrame(np.squeeze(pred)).transpose()
df_submission.rename(columns=dict(zip([str(i) for i in range(12)], label_cols)))
df_submission["StudyInstanceUID"] = test_files

# SUBMISSION FILE
df_submission.to_csv("submission.csv", index=False)

# GRAPH
epochs = range(1,num_epochs)
plt.plot(history.history['loss'], label='Training Set')
plt.plot(history.history['val_loss'], label='Validation Data)')
plt.title('Training and Validation loss')
plt.ylabel('MAE')
plt.xlabel('Num Epochs')
plt.legend(loc="upper left")
plt.show()
plt.savefig("loss.png")

