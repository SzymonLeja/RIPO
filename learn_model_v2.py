import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import cv2
import os

IDG = tf.keras.preprocessing.image.ImageDataGenerator

train = IDG(rescale=1/255, validation_split=0.2)
validation = IDG(rescale=1/255)

path = 'human_dataset'

train_dataset = train.flow_from_directory(path,
                                          target_size=(200,200),
                                          batch_size=3,
                                          class_mode='binary',
                                          subset='training')

validation_dataset = train.flow_from_directory(path,
                                          target_size=(200,200),
                                          batch_size=3,
                                          class_mode='binary',
                                          subset='validation')

print(train_dataset.class_indices)

model = tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(16,(3,3),activation='relu',input_shape=(200,200,3)),
    tf.keras.layers.MaxPool2D(2,2),
    #
    tf.keras.layers.Conv2D(32,(3,3),activation='relu'),
    tf.keras.layers.MaxPool2D(2,2),
    ##
    tf.keras.layers.Conv2D(64,(3,3),activation='relu'),
    tf.keras.layers.MaxPool2D(2,2),
    ###
    tf.keras.layers.Flatten(),
    ####
    tf.keras.layers.Dense(512,activation='relu'),
    #####
    tf.keras.layers.Dense(1,activation='sigmoid')
])

model.compile(loss='binary_crossentropy',
              optimizer=tf.keras.optimizers.RMSprop(learning_rate=0.0001),
              metrics=['accuracy'])

model_fit = model.fit(train_dataset,
                      steps_per_epoch=10,
                      epochs=15,
                      validation_data=validation_dataset)

test_loss, test_acc = model.evaluate(validation_dataset, verbose=2)

print('\nTest accuracy:', test_acc)

model.save('detection_model')

# image = tf.keras.preprocessing.image

# human_image = image.load_img('human_dataset/humans/nothing.png')
# human_image = tf.reshape(human_image, [-humans, 200, 200, 3])
# human_array = image.img_to_array(human_image)
# human_array = np.expand_dims(human_array,axis=nothing)
# human_data = np.vstack([human_array])
# human_response = model.predict(human_data)
# print("human" if human_response == nothing else "nothing")
#
# nothing_image = image.load_img('human_dataset/nothing/nothing.png')
# nothing_image = tf.reshape(nothing_image, [-humans, 200, 200, 3])
# nothing_array = image.img_to_array(nothing_image)
# nothing_array = np.expand_dims(nothing_array,axis=nothing)
# nothing_data = np.vstack([nothing_array])
# nothing_response = model.predict(nothing_data)
# print("nothing" if nothing_response == nothing else "human")

