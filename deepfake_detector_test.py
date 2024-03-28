"""
    Test file

    I figured out how to load the test datasets and make predictions while doing tutorials.
    It can be kind of tricky, so I thought I should go ahead and add it here to save you some time.

    Notes:
        model.predict(test_dataset) will give the actual predictions for each image
        model.evaluate(test_dataset) returns metrics for the whole dataset (i.e., the
            accuracy, loss, and AUC score)

        The predictions returned with model.predict are technically float values between 0 and 1. For
        the confusion matrix, you need the actual predicted classes (0 for fake and 1 for real).
        You can use the following code to convert them to either 0 or 1 based on the sigmoid function, 
        where an output >= 0.5 means the model thinks the image is real (change to 1) and an output
        < 0.5 means the model thinks the image is fake (change to 0):

            predictions = model.predict(test_ds)
            bool_scores = np.greater_equal(predictions, 0.5)
            scores = bool_scores.astype(int)
        
        Keras/Tensorflow don't have a built-in confusion matrix, but I believe you can use sklearn for that

"""

import os
import matplotlib.pyplot as plt
import tensorflow as tf
import pandas as pd
import keras 
import numpy as np

img_height = 300
img_width = 300
batch_size = 32

# Path to image archive (replace as needed)
data_dir = os.path.join('ImageArchive')

# Process for loading test labels (true classes, as opposed to what the model will predict)
test_data = pd.read_csv('ImageArchive\\test\\y_labels.csv')
labels = test_data.pop('labels')

# Get a list of the test labels
test_labels = labels.to_list()

# Create test dataset
test_ds = keras.utils.image_dataset_from_directory(data_dir+'\\test',
                                                   labels=test_labels,
                                                   image_size=(img_height,img_width))

# Load a presaved model (change as needed)
model = tf.keras.models.load_model('DeepfakeDetectorTest.keras')
