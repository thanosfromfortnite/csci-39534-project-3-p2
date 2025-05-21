"""
    Erika Lin
    erika.lin25@myhunter.cuny.edu
    CSCI 39534 Project 3 Part 2 - Retinal Fundus Images
    Student 3 - Sauvola Thresholding

"""

import os
import cv2
import numpy as np
import math
import matplotlib.pyplot as plt

def preprocess_image(image):
    # Gaussian
    filter = np.ones((3, 3), np.float64) / 9
    rows, cols = image.shape
    gaussian = np.zeros_like(image, dtype=np.float64)

    for i in range(1, rows - 1):
        for j in range(1, cols - 1):
            region = image[i - 1:i + 2, j - 1:j + 2]
            gaussian[i, j] = np.sum(region * filter)

    gaussian = np.uint8(np.clip(gaussian, 0, 255))

    # Histogram Equalization
    histogram = np.zeros(256, dtype=int)
    for i in range(rows):
        for j in range(cols):
            intensity = gaussian[i, j]
            histogram[intensity] += 1

    num_pixels = rows * cols
    normalized_histogram = histogram / num_pixels
    cdf = np.cumsum(normalized_histogram)
    L = 256

    equalized = np.zeros_like(gaussian, dtype=np.uint8)
    for i in range(rows):
        for j in range(cols):
            equalized[i, j] = int(cdf[gaussian[i, j]] * (L - 1))

    # Median
    filtered = equalized.copy()
    for i in range(1, rows - 1):
        for j in range(1, cols - 1):
            window = equalized[i - 1:i + 2, j - 1:j + 2]
            filtered[i, j] = np.median(window)

    return filtered

def calculate_psnr(original, processed):
    mse = np.mean((original.astype(np.float64) - processed.astype(np.float64)) ** 2)
    if mse == 0:
        return float('inf')
    PIXEL_MAX = 255.0
    psnr = 20 * math.log10(PIXEL_MAX / math.sqrt(mse))
    return psnr

def sauvola_thresholding(image, k, R, window_size):
    rows, cols = image.shape
    half_w = window_size // 2
    binarized_image = np.zeros_like(image, dtype=np.uint8)

    for i in range(rows):
        for j in range(cols):
            rmin = max(i - half_w, 0)
            rmax = min(i + half_w + 1, rows)
            cmin = max(j - half_w, 0)
            cmax = min(j + half_w + 1, cols)

            local_window = image[rmin:rmax, cmin:cmax]
            local_mean = np.mean(local_window)
            local_std = np.std(local_window)

            threshold = local_mean * (1 + k * ((local_std / R) - 1))

            if image[i, j] > threshold:
                binarized_image[i, j] = 255
            else:
                binarized_image[i, j] = 0

    return binarized_image

def main():
    image_dir = "images/grayscale"
    output_dir = "student-3"
    os.makedirs(output_dir, exist_ok=True)

    k = 0.1
    R = 128
    window_size = 7

    for filename in os.listdir(image_dir):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff')):
            img_path = os.path.join(image_dir, filename)
            image = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)

            if image is None:
                print(f"Warning: Could not read {filename}. Skipping.")
                continue

            preprocessed_img = preprocess_image(image)
            binary_img = sauvola_thresholding(preprocessed_img, k, R, window_size)

            psnr_value = calculate_psnr(image, binary_img)
            print(f"PSNR for {filename}: {psnr_value:.2f} dB")

            '''
            out_path = os.path.join(output_dir, f"ST_{filename}")
            cv2.imwrite(out_path, binary_img)
            '''

    print("Processing completed for all images.")

if __name__ == "__main__":
    main()