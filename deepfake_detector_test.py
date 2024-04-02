"""
    Test file

    I figured out how to load the test datasets and make predictions while doing tutorials.
    It can be kind of tricky, so I thought I should go ahead and add it here to save you some time.

    Notes:
        model.predict(test_dataset) will give the actual predictions for each image
        model.evaluate(test_dataset, verbose=0) returns metrics for the whole dataset (i.e., the
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
from sklearn.metrics import accuracy_score, roc_auc_score, log_loss, ConfusionMatrixDisplay

img_height = 256
img_width = 256
batch_size = 1000

# Path to image archive (replace as needed)
data_dir = os.path.join(r'DeepFake_Detector\ImageArchive2')

# Process for loading test labels (true classes, as opposed to what the model will predict)
test_data = pd.read_csv(data_dir+'\\y_labels.csv')
labels = test_data.pop('labels')

# Get a list of the test labels
test_labels = labels.to_list()

# Create test dataset
test_ds = keras.utils.image_dataset_from_directory(data_dir+'\\test',
                                                   labels=test_labels,
                                                   image_size=(img_height,img_width),
                                                   batch_size=batch_size,
                                                   shuffle=False)

# Load a presaved model (change as needed)
model_1 = tf.keras.models.load_model(r'DeepFake_Detector\DeepfakeDetector1.keras')
model_3 = tf.keras.models.load_model(r'DeepFake_Detector\DeepfakeDetector3.keras')
model_7 = tf.keras.models.load_model(r'DeepFake_Detector\DeepfakeDetector7.keras')
model_15 = tf.keras.models.load_model(r'DeepFake_Detector\DeepfakeDetector15.keras')
model_30 = tf.keras.models.load_model(r'DeepFake_Detector\DeepfakeDetector30.keras')

models = [model_1, model_3, model_7, model_15, model_30]
model_names = ['Model 1', 'Model 2', 'Model 3', 'Model 4', 'Model 5']
accuracy_scores = []
auc_scores = []
loss_scores = []

# Go through list of models and prints metrics (Accuracy, AUC score, & confusion matrix) for each
for model, name in zip(models, model_names):
    # Convert test dataset back to numpy array for evaluation
    test_images = []
    batch_labels = []
    for images, labels in test_ds.unbatch().batch(batch_size):
        test_images.append(images.numpy())
        batch_labels.append(labels.numpy())
    test_images = np.concatenate(test_images, axis=0)
    batch_labels = np.concatenate(batch_labels, axis=0)

    # Make predictions using the current model
    y_pred = model.predict(test_images)
    y_pred_bool = np.argmax(y_pred, axis=1)

    # Turns prediction values into 0 = fake, 1 = real for the confusion matrix
    bool_scores = np.greater_equal(y_pred , 0.5)
    scores = bool_scores.astype(int)

    # Visualizer for the confusion matrix 
    ConfusionMatrixDisplay.from_predictions(test_labels, scores, 
                                        display_labels=np.unique(test_labels), 
                                        cmap='Blues', normalize=None)

    plt.title('Confusion Matrix for {}'.format(name))
    plt.xlabel('Predicted labels')
    plt.ylabel('True labels')
    plt.show()
    
    # Calculate Accuracy, AUC, Loss and puts metrics into specified arrays
    accuracy_scores.append(accuracy_score(test_labels, scores))
    auc_scores.append(roc_auc_score(test_labels, y_pred))
    loss_scores.append(log_loss(test_labels, y_pred))

# Print the scores in an array
print(f"Classifier Model Performance Metrics:")
print(f"Accuracy: {accuracy_scores}")
print(f"AUC: {auc_scores}")
print(f"Loss: {loss_scores}\n")

# Plotting
plt.figure(figsize=(10, 7))

offset = 0.01

# Plot all metrics in a graph
plt.plot(model_names, accuracy_scores, marker='o', linestyle='-', label='Accuracy', color='blue')
plt.plot(model_names, auc_scores, marker='o', linestyle='-', label='AUC', color='green')
plt.plot(model_names, loss_scores, marker='o', linestyle='-', label='Loss', color='red')

for i, (x, y1, y2, y3) in enumerate(zip(model_names, accuracy_scores, auc_scores, loss_scores)):
    plt.text(i, y1+offset, f'{y1:.5f}', ha='center', va='bottom', color='blue')
    plt.text(i, y2+offset, f'{y2:.5f}', ha='center', va='bottom', color='green')
    plt.text(i, y3+offset, f'{y3:.5f}', ha='center', va='bottom', color='red')

plt.xlabel('Model Number')
plt.title('Model Performance')
plt.legend()
plt.grid(True)
plt.tight_layout()

plt.show()

