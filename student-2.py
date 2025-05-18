"""
    Jesse Han
    jesse.han53@myhunter.cuny.edu
    CSCI 39534 Group Project Part 2, Student 2
    Resources: Kaggle, for Images
               https://www.kaggle.com/datasets/kssanjaynithish03/retinal-fundus-images/data
               
"""

from PIL import Image
import math
import numpy as np
import os
import time

image_dir = "images/grayscale"
output_dir = "student-2"

# Sourced from G_Locally_Adaptive_Thresholding.m
def adaptive_local_thresholding(image, k=0.5, window_size=7):
    out = image.copy()
    out_pxl = out.load()
    pxl = image.load()

    width = image.size[0]
    height = image.size[1]

    for i in range(width):
        for j in range(height):
            local_max = 0
            local_min = 255
            local_mean = 0
            local_population = 0

            # Setting dimensions for the local window
            x_min = max(i - (window_size//2), 0)
            x_max = min(i + 1 + (window_size//2), width - 1)
            y_min = max(j - (window_size//2), 0)
            y_max = min(j + 1 + (window_size//2), height - 1)
            
            for x in range(x_min, x_max):
                for y in range(y_min, y_max):
                    local_max = max(local_max, pxl[x,y])
                    local_min = min(local_min, pxl[x,y])
                    local_mean += pxl[x,y]
                    local_population += 1
            local_mean = local_mean // local_population

            T = k * (local_mean + ((local_max - local_min)/255.0))

            if pxl[i,j] > T:
                out_pxl[i,j] = 255
    return out

# Find every image file name in a dir
start_time = time.time()
image_file_names = os.listdir(image_dir)

# Apply the filters and thresholding to each image
for image_file_name in image_file_names:
    image = Image.open(f"{image_dir}/{image_file_name}")

    alt_image = adaptive_local_thresholding(image)
    alt_image.save(f"{output_dir}/ALT_{image_file_name}")
    alt_image.close()
    image.close()
print(f"Process took: {time.time() - start_time} seconds.")
