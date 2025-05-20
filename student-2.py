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

'''
    Contrast Enhancements
'''

# Reused from project part 1
def linear_contrast_stretching(image_obj, t=127):
    image = image_obj.copy()
    pixels = image.load()
    width = image.size[0]
    height = image.size[1]
    output = Image.new(mode='L', size=(width, height))
    output_pixels = output.load()
    i_min = 255
    i_max = 0

    for i in range(width):
        for j in range(height):
            i_min = min(i_min, pixels[i,j])
            i_max = max(i_max, pixels[i,j])

    for i in range(width):
        for j in range(height):
            if pixels[i,j] > t:
                output_pixels[i,j] = (int) ((pixels[i,j] - t + 1) * ((float) (255 - t + 1) / (i_max - t + 1)) + t + 1)
            else:
                if (t - i_min) == 0:
                    output_pixels[i,j] = (int) ((pixels[i,j] - i_min) * ((float) (t)))
                else:
                    output_pixels[i,j] = (int) ((pixels[i,j] - i_min) * ((float) (t) / (t - i_min)))
    return output

# Reused from Lab 4
def histogram_equalization(image):
    img = image.copy()
    pixels = img.load()
    
    histogram = [0] * 256
    width = img.size[0]
    height = img.size[1]
    
    # Creating frequencies
    for i in range(width):
        for j in range(height):
            histogram[pixels[i,j]] += 1

    # Normalizing histogram by reducing the frequencies so that they add up to 1
    reduction = sum(histogram)
    histogram = [x / float(reduction) for x in histogram]

    cdf = histogram
    
    for i in range(1, len(cdf)):
        cdf[i] = cdf[i] + cdf[i-1]

    for i in range(width):
        for j in range(height):
            pixels[i,j] = (int) (255 * cdf[pixels[i,j]])

    return img

'''
    Edge Detection Operators
'''
sobelX = (
    (-1, 0, 1),
    (-2, 0, 2),
    (-1, 0, 1)
)
sobelY = (
    (1, 2, 1),
    (0, 0, 0),
    (-1, -2, -1)
)

def apply_edge_convolution(image, filterX, filterY):
    img = image.copy()
    pxl = img.load()
    width = img.size[0]
    height = img.size[1]
    filter_width = len(filterX)
    filter_height = len(filterX[0])
    
    output = Image.new(mode='L', size=(width, height))
    output_pixels = output.load()
    
    for i in range(width):
        for j in range(height):
            Gx = 0
            Gy = 0
            # Range values are centered around [i,j] based on filter size
            # Creating offsets based on index sizes
            x_from = (filter_width - 1) // -2
            x_to = (filter_width - 1) // 2 + 1
            y_from = (filter_height - 1) // -2
            y_to = (filter_height - 1) // 2 + 1
            for x in range(x_from, x_to):
                for y in range(y_from, y_to):
                    if i+x >= 0 and i+x < width and j+y >= 0 and j+y < height:
                        # Reversing the offsets since the filters are indexed from 0, not the negative offset
                        Gx += pxl[i+x,j+y] * filterX[-y_from + y][-x_from + x]
                        Gy += pxl[i+x,j+y] * filterY[-y_from + y][-x_from + x]
            G = (int) (math.sqrt(Gx ** 2 + Gy ** 2))
            output_pixels[i,j] = G
    img.close()
    return output

'''
    Bitplane Decomposition
'''

# Drops the lowest given number of bitplanes on an image
def bitplane_reconstruction(image, drop=5):
    img = image.convert('L')
    width = img.size[0]
    height = img.size[1]
    pxl = img.load()
    
    out = Image.new(mode='L', size=(width, height))
    out_pxl = out.load()
    
    for i in range(width):
        for j in range(height):
            current_pxl = pxl[i,j]
            intensity = 0
            for k in range(7, -1 + drop, -1):
                if current_pxl >= (2 ** k):
                    current_pxl -= (2 ** k)
                    intensity += (2 ** k)
            out_pxl[i,j] = intensity
    return out

'''
    Filters
'''
gaussian_filter = [
        [1.0 / 16.0, 2.0 / 16.0, 1.0 / 16.0],
        [2.0 / 16.0, 4.0 / 16.0, 2.0 / 16.0],
        [1.0 / 16.0, 2.0 / 16.0, 1.0 / 16.0]
    ]

def apply_filter_2d(image, filter_2d):
    img = image.convert('L')
    width = image.size[0]
    height = image.size[1]
    output = Image.new(mode='L', size=(width,height))
    image_pixels = image.load()
    output_pixels = output.load()

    for i in range(width):
        for j in range(height):
            filter_sum = 0.0
            for k in range(len(filter_2d)):
                for l in range(len(filter_2d[0])):
                    if i + k >= 0 and i + k < width and j + l >= 0 and j + l < height:
                        filter_sum += image_pixels[i + k, j + l] * filter_2d[k][l]
            output_pixels[i,j] = (int) (filter_sum)
    return output

'''
    Main
'''
# Find every image file name in a dir
start_time = time.time()
image_file_names = os.listdir(image_dir)
optimal_t = [103, 183, 112, 97, 97, 121, 112, 74, 91, 101, 115]

# Apply the filters and thresholding to each image
for image_file_name, t in zip(image_file_names, optimal_t):
    image = Image.open(f"{image_dir}/{image_file_name}")

    image = linear_contrast_stretching(image, t)

    image = histogram_equalization(image)
    
    image = bitplane_reconstruction(image)

    image = apply_filter_2d(image, gaussian_filter)
    
    alt_image = adaptive_local_thresholding(image, 0.7)
    alt_image.save(f"{output_dir}/ALT_{image_file_name}")
    
    alt_image.close()
    image.close()
    print(f"{image_file_name} done!")
print(f"Process took: {time.time() - start_time} seconds.")
