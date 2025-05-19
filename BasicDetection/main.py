import cv2
import os
import glob
import torch
import numpy as np
from detector import cascadeDetectEarVJ, detectEarYOLOV5, calculateIOU, calculateAccuracy
from files import loadImage, loadAnnotation

left_ear_cascade = None
right_ear_cascade = None
model = None

# Initializing the VJ cascade classifier and the YOLOv5 model
def setClassifiers():
    global left_ear_cascade, right_ear_cascade, model
    left_ear_cascade = cv2.CascadeClassifier("haarcascade_mcs_leftear.xml")
    right_ear_cascade = cv2.CascadeClassifier("haarcascade_mcs_rightear.xml")  

    weights = os.path.join(os.path.dirname(os.path.realpath(__file__)), '', 'yolo5s.pt')
    model = torch.hub.load('ultralytics/yolov5', 'custom', path=weights)

# Detection on a chosen image + image example
def detectSingleImage(imageName, scaleFactor, minNeighbors, iouValuesVJ, iouValuesYOLOV5):
    print("Detection for image " + os.path.basename(imageName) + "...")
    # Load images for presenting
    imageOriginal1 = cv2.imread(imageName)
    imageOriginal2 = cv2.imread(imageName)

    # Load image for detection
    image, imageWidth, imageHeight = loadImage(imageName) 
    # Load ground-truth
    annotation = loadAnnotation(imageName, imageWidth, imageHeight)

    # VJ detection
    detectedVJ = cascadeDetectEarVJ(image, scaleFactor, minNeighbors, left_ear_cascade, right_ear_cascade)
    # YOLOv5 detection
    detectedYOLOV5 = detectEarYOLOV5(imageName, imageWidth, imageHeight, model)

    # Calculating VJ and YOLOv5 IoU for different thresholds (0.01-1.00, step 0.01)
    for i in range(1,101):
        iouVJ = calculateIOU(annotation, detectedVJ, float(i/100))
        iouYOLOV5 = calculateIOU(annotation, detectedYOLOV5, float(i/100))
        
        iouValuesVJ[i-1] = iouVJ
        iouValuesYOLOV5[i-1] = iouYOLOV5

    # Calculating simple accuracy for VJ and YOLOv5
    accuracyVJ = calculateAccuracy(annotation, detectedVJ)
    accuracyYOLOV5 = calculateAccuracy(annotation, detectedYOLOV5)

    # Drawing VJ prediction boxes on the first image
    for (column, row, width, height) in detectedVJ:
        cv2.rectangle(imageOriginal1,(column, row),(column + width, row + height),(0, 0, 255),4)   
    # Drawing ground-thruth box on both images
    for (column, row, width, height) in [annotation]:
        cv2.rectangle(imageOriginal1,(column, row),(column + width, row + height),(0, 255, 0),4)    
        cv2.rectangle(imageOriginal2,(column, row),(column + width, row + height),(0, 255, 0),4)    
    # Drawing YOLOv5 prediction boxed on the second image
    for (column, row, width, height) in detectedYOLOV5:
        cv2.rectangle(imageOriginal2,(column, row),(column + width, row + height),(0, 0, 255),4)    

    print("---RESULTS---")
    print("Ground-truth: ", annotation)
    print("Detected VJ: ", detectedVJ)
    print("Detected YOLOv5: ", detectedYOLOV5)
    print("Accuracy VJ: ", accuracyVJ)
    print("Accuracy YOLOv5: ", accuracyYOLOV5)
    print("IoUs VJ: ", iouValuesVJ)
    print("IoUs YOLOv5: ", iouValuesYOLOV5)
    print("Predictions VJ: ", len(detectedVJ))
    print("Predictions YOLOv5: ", len(detectedYOLOV5))
    print("---END---")

    # Show both pictures with the detection boxes on them
    # The pictures are resized to 800x800 to be visible easier
    cv2.imshow("Viola-Jones/Haar cascade detection", cv2.resize(imageOriginal1, (800, 800)))
    cv2.imshow("YOLOv5 detection", cv2.resize(imageOriginal2, (800, 800)))
    cv2.waitKey(0)



if __name__ == '__main__':
    # First parameter for VJ detection (1.01, 1.03, 1.05)
    scaleFactor = 1.05

    # Second parameter for VJ detection (1, 3, 5)
    minNeighbors = 3

    # If single mode is true the program will do detection on one chosen photo only
    # Else it will do the detection on all 500 images evaluate results 
    singleMode = True

    # Initialize the detection methods
    setClassifiers()

    # IoU values of the VJ detection and the YOLOv5 detection for all thresholds
    iouValuesVJ = np.zeros(100)
    iouValuesYOLOV5 = np.zeros(100)

    # Single mode
    if singleMode:
        # Do the VJ and YOLOv5 detection of the chosen photo and evaluate the results
        detectSingleImage("ear_data/test/2181.png", scaleFactor, minNeighbors, iouValuesVJ, iouValuesYOLOV5)
    # Not single mode
    else:
        # Names of images that have no VJ detections
        imageNamesOfNoDetectedVJ = []
        # The best VJ accuracies
        bestAccuracyVJ = []
        # Simple accuracy scores for VJ and YOLOv5
        accuracyScoreVJ = []
        accuracyScoreYOLOV5 = []
        # Names of images with YOLO prediction that have more than one detection
        imageNamesOfMoreThanOneDetectedYOLOV5 = []
        # False positive preditiction for VJ and YOLOv5
        falsePositivesVJ = np.zeros((100,), dtype=int)
        falsePositivesYOLOV5 = np.zeros((100,), dtype=int)
        # True positive preditiction for VJ and YOLOv5
        truePositivesVJ = np.zeros((100,), dtype=int)
        truePositivesYOLOV5 = np.zeros((100,), dtype=int)
        # (image name, iou, VJ predictions) for each image for VJ prediction with best iou
        bestVJPredictionImages = []
        # (image name, VJ predictions) for each image for VJ prediction with iou = 0.0 and predictions > 0
        failedVJPredictionImages = []
        # Shows the current image that is evaluated
        counter = 1

        # Go thru all images
        for imageName in glob.glob("ear_data/test/*.png"):
            print("Detection for image " + str(counter) + "...")
            # Load image
            image, imageWidth, imageHeight = loadImage(imageName)
            # Load ground-truth
            annotation = loadAnnotation(imageName, imageWidth, imageHeight)

            # Do the VJ and the YOLOv5 detection for the current image
            detectedVJ = cascadeDetectEarVJ(image, scaleFactor, minNeighbors, left_ear_cascade, right_ear_cascade)
            detectedYOLOV5 = detectEarYOLOV5(imageName, imageWidth, imageHeight, model)

            # Calculating VJ and YOLOv5 IoU for different thresholds (0.01-1.00, step 0.01)
            for i in range(1,101):
                # Calculate VJ and YOLOv5 IoU for the current threshold
                iouVJ = calculateIOU(annotation, detectedVJ, float(i/100))
                iouYOLOV5 = calculateIOU(annotation, detectedYOLOV5, float(i/100))
                
                if iouVJ > 0.0:
                    if i == 1:
                        bestVJPredictionImages.append((os.path.basename(imageName).split(".")[0], iouVJ, len(detectedVJ)))
                    truePositivesVJ[i-1] += 1
                    iouValuesVJ[i-1] += iouVJ
                    falsePositivesVJ[i-1] += len(detectedVJ) - 1
                else:
                    if len(detectedVJ) != 0:
                        if i == 1:
                            failedVJPredictionImages.append((os.path.basename(imageName).split(".")[0], len(detectedVJ)))
                        falsePositivesVJ[i-1] += len(detectedVJ)

                if iouYOLOV5 > 0.0:
                    truePositivesYOLOV5[i-1] += 1
                    iouValuesYOLOV5[i-1] += iouYOLOV5
                    falsePositivesYOLOV5[i-1] += len(detectedYOLOV5) - 1
                else:
                    if len(detectedYOLOV5) != 0:
                        falsePositivesYOLOV5[i-1] += len(detectedYOLOV5)

            # If YOLOv5 detection has more than one prediction
            if (len(detectedYOLOV5) > 1):
               imageNamesOfMoreThanOneDetectedYOLOV5.append(os.path.basename(imageName).split(".")[0])
            # If VJ detection has no predictions
            if (len(detectedVJ) == 0):
                imageNamesOfNoDetectedVJ.append(os.path.basename(imageName).split(".")[0])

            # Calculating simple accuracy for VJ and YOLOv5 for the current image
            accuracyVJ = calculateAccuracy(annotation, detectedVJ)
            accuracyYOLOV5 = calculateAccuracy(annotation, detectedYOLOV5)

            if (accuracyVJ != 0.0):
                bestAccuracyVJ.append((os.path.basename(imageName).split(".")[0], accuracyVJ))

            accuracyScoreVJ.append(accuracyVJ)
            accuracyScoreYOLOV5.append(accuracyYOLOV5)
            counter += 1

        # Precisions for VJ and YOLOv5 detections for all images over all thresholds
        precisionsVJ = []
        precisionsYOLOV5 = []
        # Recalls for VJ and YOLOv5 detections for all images over all thresholds
        recallsVJ = []
        recallsYOLOV5 = []
        # IoU averages for VJ and YOLOv5 detections for all images over all thresholds
        iouAveragesVJ = []
        iouAveragesYOLOV5 = []
        
        for i in range(0, 100):
            precisionsVJ.append(truePositivesVJ[i] / (truePositivesVJ[i] + falsePositivesVJ[i]))
            recallsVJ.append(truePositivesVJ[i] / 500)
            iouAveragesVJ.append(iouValuesVJ[i] / 500)
            precisionsYOLOV5.append(truePositivesYOLOV5[i] / (truePositivesYOLOV5[i] + falsePositivesYOLOV5[i]))
            recallsYOLOV5.append(truePositivesYOLOV5[i] / 500)
            iouAveragesYOLOV5.append(iouValuesYOLOV5[i] / 500)

        print("---RESULTS---")
        print("Precisions VJ: ", precisionsVJ)
        print("Recalls VJ: ", recallsVJ)
        print("Average IoUs VJ: ", iouAveragesVJ)
        print("Precisions YOLOv5: ", precisionsYOLOV5)
        print("Recalls YOLOv5: ", recallsYOLOV5)
        print("Average IoUs YOLOv5: ", iouAveragesYOLOV5)
        print("Empty VJs: ", imageNamesOfNoDetectedVJ)
        print("VJs with best accuracy: ", bestAccuracyVJ)
        print("Accuracy score VJ: ", np.average(accuracyScoreVJ))
        print("More than one detections YOLOv5: ", imageNamesOfMoreThanOneDetectedYOLOV5)
        print("Accuracy score YOLOv5: ", np.average(accuracyScoreYOLOV5))
        print("Best predictions VJ (image, IoU, predictions): ", sorted(bestVJPredictionImages, key=lambda tup: tup[1], reverse=True))
        print("Failed predictions VJ (image, predictions): ",   sorted(failedVJPredictionImages, key=lambda tup: tup[1], reverse=True))
        print("---END---")
        # Use these results in drawPlots.py to draw the plots