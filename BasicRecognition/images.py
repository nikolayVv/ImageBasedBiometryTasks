import glob
import os
import cv2
import numpy as np
import math
from skimage import feature

# Loads all 1000 images, resize them, grayscale them and put them in array, which is returned by the function
# Class 1 -> samples[0] - samples[9]
# Class 2 -> samples[10] - samples [19]
# ...
# Class 100 -> samples[990] - samples[999]
def loadAndConvertImages2D(path, newDim, newColor):
    images2D = []
    for className in os.listdir(path):
        if os.path.isdir(path + '/' + className):
            for file in glob.glob(path + "/" + className + "/*.png"):
                image = cv2.imread(file)
                resizedImage = cv2.resize(image, newDim, interpolation = cv2.INTER_AREA)
                newImage = cv2.cvtColor(resizedImage, newColor)
                images2D.append(newImage)
    return images2D

# Converts a 2D array into a 1D array
def convertTo1D(samples):
    samples1D = []
    samplesHist1D = []
    for sample in samples:
        sample = np.concatenate(sample)
        samples1D.append(sample)
        histogram = calculateHistogram(sample, 256)
        samplesHist1D.append(histogram)
    return samples1D, samplesHist1D

# Checks if two indexes of photos are from the same class (folder number)
# All possible classes are from 1 to 100
def haveSameClass(indexImageA, indexImageB):
    return math.floor(indexImageA / 10) == math.floor(indexImageB / 10)

# Creates a LBP array, that has 2D arrays for each image using the already implemented Scikit
def convertToScikitLBP1D(samples, radius, codeLength, type):
    imagesLBP1D = []
    imagesLBPHist1D = []

    for sample in samples:
        imageLBP = feature.local_binary_pattern(sample, codeLength, radius, method=type)
        imagesLBP1D.append(np.concatenate(imageLBP))
        if type == "uniform":
            histogram = calculateHistogram(imageLBP, codeLength + 1)
        else:
            n_bins = int(math.pow(2, codeLength))
            histogram = calculateHistogram(imageLBP, n_bins)
        imagesLBPHist1D.append(histogram)
    return imagesLBP1D, imagesLBPHist1D

# Creates a LBP array, that has 2D arrays for each image (custom created LBP method)
def convertToLBP1D(samples, height, width, radius, codeLength, type):
    imagesLBP1D = []
    imagesLBPHist1D = []
    for k in range(len(samples)):
        imageLBP = []
        for i in range(0, height):
            for j in range(0, width):
                value = calculateLBP(samples[k], i, j, radius, codeLength, type)
                imageLBP.append(value)
        imagesLBP1D.append(imageLBP)
        if type == "uniform":
            histogram = calculateHistogram(imageLBP, codeLength + 1)
        else:
            n_bins = int(math.pow(2, codeLength))
            histogram = calculateHistogram(imageLBP, n_bins)
        imagesLBPHist1D.append(histogram)
    return imagesLBP1D, imagesLBPHist1D

# Calculates the LBP value for each pixel from an image (custom created LBP method)
def calculateLBP(sample, x, y, radius, codeLength, type):
    center = sample[x][y];
    imageLBP = []

    for i in range(-radius, radius+1):
            if i == radius or i == -radius:
                for j in range(-radius, 1):
                    if (j == 0):
                        imageLBP.append(getPixelValue(sample, center, 0, i))
                    else:
                        imageLBP.append(getPixelValue(sample, center, j, i))
                        imageLBP.append(getPixelValue(sample, center, -j, i))
            else:
                imageLBP.append(getPixelValue(sample, center, -radius, i))
                imageLBP.append(getPixelValue(sample, center, radius, i))
    val = 0
    if type == "uniform":
        uniformVal = abs(imageLBP[codeLength - 1] - imageLBP[0])
        for i in range(1, codeLength):
            uniformVal += abs(imageLBP[i] - imageLBP[i-1])
        if uniformVal <= 2:
            for i in range(codeLength):
                val += imageLBP[i]
        else:
            val = codeLength + 1
    else:
        for i in range(codeLength):
            val += imageLBP[i] * math.pow(2, i)
    return val

# Returns the value of the pixel up to that if it is bigger or smaller than the center (custom created LBP method)
# It is used in the calculateLBP function to change the value of the pixel to 0 or 1, which is used in the calculated LBP value later
def getPixelValue(img, center, x, y):
    value = 0
    try:
        if img[x][y] >= center:
            value = 1
    except:
        pass
    return value

# Calculates histograms for each for a 2d array of photos
# Converts the calculated 2d array of histograms to 1D array
def calculateHistogram(sample, n_bins):
    histogram, _ = np.histogram([sample], density=True, bins=n_bins, range=(0, n_bins))
    histogram = histogram.astype("float")
    histogram /= (histogram.sum() + 1e-7)
    return histogram