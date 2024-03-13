import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image
import os
import random

# Read from the CSV file
df = pd.read_csv('faces.csv')
# randomize 
df = df.sample(frac=1).reset_index(drop=True)

# Show first 10 images (change)
for index, row in df.head(10).iterrows():
    image_name = row['image_name']
    width, height = row['width'], row['height']
    x0, y0, x1, y1 = row['x0'], row['y0'], row['x1'], row['y1']

    # Load & display image
    current_directory = "main.py"
    image_path = os.path.join(current_directory, '/Users/torivers/Desktop/data/images', image_name)
    image = Image.open(image_path)

    # Show figure and axes
    fig, ax = plt.subplots()

    # Display image
    ax.imshow(image)

    # rectangle
    rect = patches.Rectangle((x0, y0), x1 - x0, y1 - y0, linewidth=2, edgecolor='r', facecolor='none')

    # Add the rectangle to the axes
    ax.add_patch(rect)

    # Show the plot
    plt.show()
