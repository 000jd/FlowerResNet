import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
from utils import classes
from keras import Sequential
from keras.layers import Dense, Flatten, Dropout
from keras.applications import ResNet50V2


def makemodel():
    base_model = ResNet50V2(weights=None,
                            include_top=False,
                            input_shape=(224, 224, 3),
                            pooling='max',)

    for index, layer in enumerate(base_model.layers):
        if index < 80:
            layer.trainable = False
        else:
            layer.trainable = True

    resnet_model = Sequential()

    resnet_model.add(base_model)
    resnet_model.add(Flatten())
    resnet_model.add(Dropout(0.3))
    resnet_model.add(Dense(299, activation='softmax'))

    resnet_model.load_weights('tfmodel-weights.h5')

    return resnet_model


def tf_predict_flower(image):
    image = np.array(image)  # Convert image to a NumPy array
    image = tf.image.resize(image, (224, 224))
    image = tf.image.convert_image_dtype(image, tf.float32)
    image = tf.expand_dims(image, axis=0)

    model = makemodel()

    pred = model.predict(image)
    output_class = classes.class_names[np.argmax(pred)]
    return output_class
