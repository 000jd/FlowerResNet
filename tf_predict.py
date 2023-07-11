import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow import keras
import numpy as np
from utils import classes
import cv2


def tf_predct_flower(image):

    image = cv2.resize(image, (224, 224))
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = np.expand_dims(image, axis=0)
    image = tf.cast(image/255., tf.float32)

    model = tf.keras.models.load_model('./models/model_L_0.17vl_0.52.h5')

    pred = model.predict(image)
    output_class = classes.class_names[np.argmax(pred)]
    return output_class
