import pandas as pd
import matplotlib.pyplot as plt
from scikit import transform
from PIL import Image, ImageFilter, ImageDraw
import os
import random
import numpy as np

def warp_image(image, keypoints, warp_strength=0.1, blur_radius = 2):
 
    width, height = image.size
    new_keypoints = [(x + random.uniform(-warp_strength, warp_strength) * width, 
                      y + random.uniform(-warp_strength, warp_strength) * height) 
                     for x, y in keypoints]
    
    # face mapping
    mask = Image.new('L', (width, height), 0)
    draw = ImageDraw.Draw(mask)
    draw.polygon(new_keypoints, fill=255, outline=255)  # Fill and outline the polygon
    
    # blur within polygon
    blurred_image = image.filter(ImageFilter.GaussianBlur(blur_radius))
    warped_image = Image.composite(blurred_image, image, mask)  # Swap the order of images

    return warped_image

# reading from faces.csv file
df = pd.read_csv('faces.csv')
df = df.sample(frac=1).reset_index(drop=True) # randomizing (can remove)

# Number of images to warp and save
num_images_to_warp = 500

# creating a directory to store warped images
warped_images_dir = 'warped_images'
if not os.path.exists(warped_images_dir):
    os.makedirs(warped_images_dir)

# Iterate over each row in the dataframe & limit to the num of images
for index, row in df.head(num_images_to_warp).iterrows():
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

    # Save warped image
    warped_image_path = os.path.join(warped_images_dir, f'warped_{image_name}')
    warped_image.save(warped_image_path)

    print(f"Warped image saved: {warped_image_path}")
