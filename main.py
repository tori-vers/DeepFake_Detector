import pandas as pd
import matplotlib.pyplot as plt
from scikit import transform
from PIL import Image
import os
import random

def warp_image(image, keypoints, warp_strength=0.1):
 
    width, height = image.size
    new_keypoints = [(x + random.uniform(-warp_strength, warp_strength) * width, 
                      y + random.uniform(-warp_strength, warp_strength) * height) 
                     for x, y in keypoints]
    
    # matrix 1
    src = np.array(keypoints)
    dst = np.array(new_keypoints)
    tform = transform.ProjectiveTransform()
    tform.estimate(src, dst)
    
    # Warping
    warped_image = transform.warp(np.array(image), tform.inverse)
    
    return Image.fromarray((warped_image * 255).astype(np.uint8))

# Read from the CSV file
df = pd.read_csv('faces.csv')
# randomize 
df = df.sample(frac=1).reset_index(drop=True)

# Show first 10 images w/ "alterations"
for index, row in df.head(10).iterrows():
    image_name = row['image_name']
    width, height = row['width'], row['height']
    x0, y0, x1, y1 = row['x0'], row['y0'], row['x1'], row['y1']

    # Load img
    current_directory = "main.py"
    image_path = os.path.join(current_directory, '/Users/torivers/Desktop/data/images', image_name)
    image = Image.open(image_path)

    # Warp img
    keypoints = [(x0, y0), (x1, y0), (x1, y1), (x0, y1)]
    warped_image = warp_image(image, keypoints)

    # Show original & warped 
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    axes[0].imshow(image)
    axes[0].set_title('Original')
    axes[1].imshow(warped_image)
    axes[1].set_title('Warped')

    plt.show()

