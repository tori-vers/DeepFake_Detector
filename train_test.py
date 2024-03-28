import os
import matplotlib.pyplot as plt
import tensorflow as tf
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

# Custom data augmentation layer (to prevent overfitting)
data_augmentation = tf.keras.Sequential(
    [
        layers.RandomFlip("horizontal", input_shape=(img_height, img_width, 3)),
        layers.RandomContrast(0.2),
        layers.RandomBrightness(0.2),
        layers.RandomRotation(0.2),
        layers.RandomZoom(0.2)
    ]
)

# Classifier model construction
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
    layers.Conv2D(64, 3, padding='same', activation='relu'),
    layers.MaxPooling2D(),
    layers.Dropout(0.2),
    layers.Conv2D(64, 3, padding='same', activation='relu'),
    layers.MaxPooling2D(),
    layers.Dropout(0.2),
    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dropout(0.5),
    layers.Dense(1, activation='sigmoid')
])

# Set optimizer, loss function, and metrics
model.compile(loss='binary_crossentropy', 
              optimizer='adam', 
              metrics=['accuracy', tf.keras.metrics.AUC()])

# Show summary of model layers and parameters
model.summary()

# Train model
history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=3
)

# Save model configuration
model.save('DeepfakeDetector.keras')