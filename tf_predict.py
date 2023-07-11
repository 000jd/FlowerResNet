import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
from utils import classes

def tf_predict_flower(image):
    image = np.array(image)  # Convert image to a NumPy array
    image = tf.image.resize(image, (224, 224))
    image = tf.image.convert_image_dtype(image, tf.float32)
    image = tf.expand_dims(image, axis=0)

    model = tf.keras.models.load_model('FlowerResNet/models/model_L_0.17vl_0.52.h5')

    pred = model.predict(image)
    output_class = classes.class_names[np.argmax(pred)]
    return output_class
