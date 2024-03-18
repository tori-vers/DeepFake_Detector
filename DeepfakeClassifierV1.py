"""
Deepfake Classifier Version 1
    Notes:
    Training/Validation image dataset is combination of images from the following 
    Kaggle datasets:
        https://www.kaggle.com/datasets/ciplab/real-and-fake-face-detection/data
        https://www.kaggle.com/datasets/undersc0re/fake-vs-real-face-classification/data?select=test
    This yields a total dataset of 3646 image files, with 2917 files used for training and 729 files
    used for validation. Images are shuffled, which results in slightly different accuracy metrics each
    time the model is trained.
    100 images from the first dataset are held back for the test phase, not used in training or validation.

    Used the following sources as guides:
        https://www.tensorflow.org/tutorials/images/classification
        https://www.kaggle.com/code/tharunnayak14/real-vs-fake-face-100-tensorflow-cnn/notebook
    The second source used a different dataset type than I did (real vs GAN generated images), but was 
    useful for creating the actual model.
    The discussion on the first source about overfitting was especially useful.

    The number of epochs is found through trial and error. At some point, the training accuracy/loss will
    keep increasing, but the validation accuracy/loss will start to decrease. When this happens, it means
    the model has overfit on the training data and will no longer be able to generalize well to new data.
    Data augmentation and dropout layers in the model help with this, as does having a good number of varied
    training images. I've found that anything less than 2500 images total for training is too little. Avoiding
    overtraining by limiting the number of epochs is also necessary.

    If training on CPU, it will take several minutes to train the model. For instance, my intel i7 cpu takes ~45
    seconds per epoch. Training is also battery intensive, so plug in if on a laptop. Because of this, and
    the fact that the model is slightly different every time it's trained, a final model will be saved that 
    can be reloaded and used for the testing phases.
"""

import os
import matplotlib.pyplot as plt
import tensorflow as tf
import pandas as pd # for loading test labels
import keras # for loading test data
from keras import layers
from keras.models import Sequential

# Path to image archive (replace as needed)
data_dir = os.path.join('ImageArchive')

img_height = 300
img_width = 300
batch_size = 32

# Training dataset (80% of total)
train_ds = tf.keras.utils.image_dataset_from_directory(
    data_dir+'\\train',
    validation_split=0.2,
    subset="training",
    seed=123,
    image_size=(img_height, img_width),
    batch_size=batch_size
)

# Validation dataset (20% of total)
val_ds = tf.keras.utils.image_dataset_from_directory(
    data_dir+'\\train',
    validation_split=0.2,
    subset="validation",
    seed=123,
    image_size=(img_height, img_width),
    batch_size=batch_size
)

# Performance configuration (buffered prefetching)
AUTOTUNE = tf.data.AUTOTUNE
train_ds = train_ds.cache().shuffle(1000).prefetch(buffer_size=AUTOTUNE)
val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)

# Custom data augmentation layer
data_augmentation = tf.keras.Sequential(
    [
        layers.RandomFlip("horizontal", input_shape=(img_height, img_width, 3))
    ]
)

# Classifier 
model = Sequential(
[
    data_augmentation,
    layers.Rescaling(1./255),
    layers.Conv2D(16, 3, padding='same', activation='relu'),
    layers.MaxPooling2D(),
    layers.Dropout(0.2), 
    layers.Conv2D(32, 3, padding='same', activation='relu'),
    layers.MaxPooling2D(),
    layers.Dropout(0.2),
    layers.Conv2D(64, 3, padding='same', activation='relu'),
    layers.MaxPooling2D(),
    layers.Dropout(0.2),
    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dropout(0.4),
    layers.Dense(1, activation='sigmoid')
])

# Show summary of model layers and parameters
model.summary()

# Number of epochs to train (15 yields accuracy around 95% and validation accuracy around 85%)
epochs = 15

# Set optimizer, loss function, and metrics
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

# Train model
history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=epochs
)

# Show chart of accuracy and loss on training and validation sets
acc = history.history['accuracy']
val_acc = history.history['val_accuracy']

loss = history.history['loss']
val_loss = history.history['val_loss']

epochs_range = range(epochs)

plt.figure(figsize=(8, 8))
plt.subplot(1, 2, 1)
plt.plot(epochs_range, acc, label='Training Accuracy')
plt.plot(epochs_range, val_acc, label='Validation Accuracy')
plt.legend(loc='lower right')
plt.title('Training and Validation Accuracy')

plt.subplot(1, 2, 2)
plt.plot(epochs_range, loss, label='Training Loss')
plt.plot(epochs_range, val_loss, label='Validation Loss')
plt.legend(loc='upper right')
plt.title('Training and Validation Loss')
plt.show()

# Save model configuration
#model.save('DeepfakeDetectorV1.keras')

"""Process for loading test data for steps 3 and 4
test_data = pd.read_csv('C:\\Users\\tsuki\\Documents\\CSCI 5530\\DeepFake Experiment Code\\archive (2)\\test\\y_labels.csv')
labels = test_data.pop('labels')
test_labels = labels.to_list()

test_ds = keras.utils.image_dataset_from_directory(data_dir+'\\test',labels=test_labels,image_size=(img_height,img_width))"""