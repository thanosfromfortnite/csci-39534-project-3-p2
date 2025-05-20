"""
    Novin Tang
    novin.tang44@myhunter.cuny.edu
    CSCI 39534 Project 3 Part 2 - Retinal Fundus Images
    Student 1 - Otsu Thresholding
"""

from PIL import Image
import numpy as np
from scipy import ndimage
import matplotlib.pyplot as plt
import os

# Default window size of 10x10
def calculateEme(image: Image.Image, k1:int = 10, k2: int = 10):
    img_arr = np.array(image).astype(np.double)
    rows, cols = img_arr.shape
    block_rows, block_cols = rows // k1, cols // k2
    eme = 0

    for i in range(k1):
        for j in range(k2):
            row_start = i * block_rows
            row_end = (i + 1) * block_rows
            col_start = j * block_cols
            col_end = (j + 1) * block_cols

            block = img_arr[row_start:row_end, col_start:col_end]
            
            I_max = np.max(block)
            I_min = np.min(block)

            if I_min == 0:
                I_min = 1

            eme += 20 * np.log(I_max / I_min)
    
    eme /= (k1 * k2)
    return eme

def calculatePsnr(original: Image.Image, noisy: Image.Image):
    original_arr = np.array(original)
    noisy_arr = np.array(noisy)
    mse = np.mean((original_arr.astype(np.float32) - noisy_arr.astype(np.float32)) ** 2)
    psnr = 10 * np.log10((255.0 ** 2) / mse)

    return psnr

def applyMedianFilter(img: Image.Image, window: tuple):
    img_arr = np.array(img)
    # Apply median filter with given tuple for window size and 0 padding
    return Image.fromarray(ndimage.median_filter(img_arr, window, mode='constant', cval=0))

def applyGaussianFilter(img: Image.Image, stdev: int):
    img_arr = np.array(img)
    # Apply Gaussian filter with provided standard deviation and 0 padding
    return Image.fromarray(ndimage.gaussian_filter(img_arr, sigma=stdev, mode='constant', cval=0))

def applyAlphaTrimMeanFilter(noisy_img):
    alpha = 2 # number of values to trim
    noisy_img_arr = np.array(noisy_img)
    rows, cols = noisy_img_arr.shape
    filtered_arr = noisy_img_arr.copy()

    for i in range(1, rows-1):
        for j in range(1, cols-1):
            region = noisy_img_arr[i-1:i+2,j-1:j+2]
            sorted_region = np.sort(region.flatten())
            trimmed_region = sorted_region[int(alpha/2):int(9-alpha/2)]
            filtered_arr[i,j] = np.mean(trimmed_region).astype(np.uint8)
    
    return Image.fromarray(filtered_arr)

def applyLinearContrastStretching(img: Image.Image):
    img_arr = np.array(img).astype(np.double)
    I_min, I_max = np.min(img_arr), np.max(img_arr)
    L_min, L_max = 0, 255

    stretched_image = np.zeros(img_arr.shape).astype(np.double)

    stretched_image = ((img_arr - I_min) / (I_max - I_min)) * (L_max - L_min) + L_min

    stretched_image = np.clip(stretched_image, 0, 255).astype(np.uint8)

    return Image.fromarray(stretched_image).convert('L')

def applyHistogramEqualization(image):
    img_arr = np.array(image)

    histogram, _ = np.histogram(img_arr, bins=256, range=(0, 255))
    rows, cols = img_arr.shape
    normalized_histogram = histogram / (rows * cols)

    cdf = np.zeros(256)
    cdf[0] = normalized_histogram[0]
    for i in range(1, len(normalized_histogram)):
        cdf[i] = cdf[i - 1] + normalized_histogram[i]

    equalized_image = np.zeros(img_arr.shape)

    equalized_image = np.clip(cdf[img_arr] * 255, 0, 255).astype(np.uint8)

    return Image.fromarray(equalized_image).convert('L')

def applySobelOperator(img: Image.Image):
    img_arr = np.array(img).astype(np.double)

    x = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=np.double)
    y = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]], dtype=np.double)

    x_result = ndimage.convolve(img_arr, x)
    y_result = ndimage.convolve(img_arr, y)

    img_result = np.sqrt(np.square(x_result) + np.square(y_result))

    return Image.fromarray(np.clip(img_result, 0, 255).astype(np.uint8)).convert('L')

def applyOtsuThresholding(img: Image.Image):
    img_arr = np.array(img)
    
    histogram, gray_levels = np.histogram(img_arr, bins=256, range=(0, 255))
    gray_levels = np.arange(256)
    norm_hist = histogram / img_arr.size

    max_between_class_variance = 0.0
    threshold = 0
    n_levels = 256
    between_class_variance = np.zeros(n_levels)
    class1_hist = np.zeros(n_levels)
    class2_hist = np.zeros(n_levels)

    for t in range(n_levels):
        # Class probabilities
        prob_class1 = norm_hist[:t+1].sum()
        prob_class2 = 1.0 - prob_class1
        class1_hist[t] = prob_class1
        class2_hist[t] = prob_class2

        # Class means
        if prob_class1 > 0:
            class1_mean = (gray_levels[:t+1] * norm_hist[:t+1]).sum() / prob_class1
        else:
            class1_mean = 0.0
        if prob_class2 > 0:
            class2_mean = (gray_levels[t+1:] * norm_hist[t+1:]).sum() / prob_class2
        else:
            class2_mean = 0.0

        # Between-class variance
        between_class_variance[t] = prob_class1 * prob_class2 * (class1_mean - class2_mean) ** 2

        # Update threshold if variance is maximized
        if between_class_variance[t] > max_between_class_variance:
            max_between_class_variance = between_class_variance[t]
            threshold = t

    segmented_image = img_arr > threshold

    # Return boolean mask
    return Image.fromarray(segmented_image).convert("L")

path = "images/grayscale"
save_path = "student-1"

def testPreprocessing():
    print("----Noise Removal Filters (PSNR)----")
    print("File | Median | Gauss  | Alpha")

    for filename in os.listdir(path):
        print('{:^5s}'.format(filename.split("_")[0]), end="|")
        image = Image.open(path + "/" + filename)
        median = applyMedianFilter(image, (3, 3))
        gauss = applyGaussianFilter(image, 2)
        alpha = applyAlphaTrimMeanFilter(image)
        print('{:^8.2f}'.format(calculatePsnr(image, median)), end="|")
        print('{:^8.2f}'.format(calculatePsnr(image, gauss)), end="|")
        print('{:^8.2f}'.format(calculatePsnr(image, alpha)))

        # Uncomment to display images

        """
        plt.figure(filename + " noise removal")
        plt.subplot(2, 2, 1)
        plt.imshow(image, cmap='gray')
        plt.title("Original Image")
        plt.subplot(2, 2, 2)
        plt.imshow(median, cmap='gray')
        plt.title("Median Filter")
        plt.subplot(2, 2, 3)
        plt.imshow(gauss, cmap='gray')
        plt.title("Gaussian Filter")
        plt.subplot(2, 2, 4)
        plt.imshow(alpha, cmap='gray')
        plt.title("Alpha Trimmed Mean Filter")
        """

    print("----Contrast Enhancement Filters (EME)----")
    print("File | Origin | Linear | Equal")

    for filename in os.listdir(path):
        print('{:^5s}'.format(filename.split("_")[0]), end="|")
        image = Image.open(path + "/" + filename)
        linear = applyLinearContrastStretching(image)
        equal = applyHistogramEqualization(image)
        print('{:^8.2f}'.format(calculateEme(image)), end="|")
        print('{:^8.2f}'.format(calculateEme(linear)), end="|")
        print('{:^8.2f}'.format(calculateEme(equal)))

        # Uncomment to display images
        """
        plt.figure(filename + "contrast_enhancement")
        plt.subplot(1, 3, 1)
        plt.imshow(image, cmap='gray')
        plt.title("Original Image")
        plt.subplot(1, 3, 2)
        plt.imshow(linear, cmap='gray')
        plt.title("Linear Contrast Stretching")
        plt.subplot(1, 3, 3)
        plt.imshow(equal, cmap='gray')
        plt.title("Histogram Equalization")
        """

# Uncomment below to test various preprocessing filters and enhancements
# testPreprocessing()

for filename in os.listdir(path):
    image = Image.open(path + "/" + filename)
    # Preprocessing step 1: Apply median filter with 3x3 window size
    st1 = applyMedianFilter(image, (3, 3))
    # Preprocessing step 2: Apply linear contrast stretching
    st2 = applyLinearContrastStretching(st1)
    # Preprocessing step 3: Apply sobel edge detection
    st3 = applySobelOperator(st2)
    # Apply Otsu thresholding technique
    st4 = applyOtsuThresholding(st3)
    # Save image to path
    st4.save(save_path + "/" + filename)

    # Uncomment to display images
    """
    plt.figure(filename)
    plt.subplot(1, 2, 1)
    plt.imshow(image, cmap='gray')
    plt.title("Original Image")
    plt.subplot(1, 2, 2)
    plt.imshow(st4, cmap='gray')
    plt.title("Enhanced Image")
    """

plt.show()