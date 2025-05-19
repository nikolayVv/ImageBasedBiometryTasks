# Image Based Biometry Tasks

A collection of three image based biometry tasks implemented in Python.

## Basic Recognition

Implementation of the Local Binary Patterns (LBP) algorithm for image recognition and conducting a thorough evaluation to assess its effectiveness. Using a straightforward pixel-wise comparison of images and benchmarking the results against OpenCV’s built-in LBP implementation. The evaluation was based on rank-1 classification accuracy, tested on datasets containing images of sizes 64x64 and 128x128 pixels. Exploring several distance metrics for feature comparison, including Euclidean, Manhattan, and Cosine distances, to understand their influence on recognition accuracy. Additionally, experimenting with different LBP configurations to examine how parameter variations affect the algorithm’s performance.

## Basic Detection

Setting up and evaluating two widely used object detection methods on a dataset of ear images. The first method is the classical Viola-Jones algorithm using Haar cascade classifiers, which is employed to predict and optimize detection performance specifically for ear localization. The second method involves configuring and running the YOLOv5 detection model on the same set of images to provide a modern, deep learning-based comparison. To evaluate and compare the results of both detection approaches, an Intersection over Union (IoU) and Precision-Recall (PR) scores were computed across all thresholds with a step size of 0.01, ensuring a fine-grained analysis of detection quality. The evaluation was conducted on a set of 500 ear images, and in addition to the overall metrics, the best and worst detection examples for each method were also identified to highlight strengths and limitations in practical scenarios.

## Iris Recognition Pipeline

Implementing a complete biometric recognition pipeline focused on iris recognition using the CASIA iris image dataset. This involved all key stages of the pipeline, including image preprocessing, iris segmentation, normalization, feature extraction, and matching. Each step was carefully implemented and optimized to ensure accurate and reliable iris recognition. The CASIA dataset provided a robust benchmark for testing the effectiveness of the pipeline, allowing for a comprehensive evaluation of the system’s performance in realistic biometric scenarios.