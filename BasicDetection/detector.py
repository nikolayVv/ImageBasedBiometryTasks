from sklearn.metrics import accuracy_score

# VJ cascade detector for the left and right ear
def cascadeDetectEarVJ(img, scaleFactor, minNeighbors, left_ear_cascade, right_ear_cascade):
    detected_left_ears = left_ear_cascade.detectMultiScale(img, scaleFactor=scaleFactor, minNeighbors=minNeighbors)
    detected_right_ears = right_ear_cascade.detectMultiScale(img, scaleFactor=scaleFactor, minNeighbors=minNeighbors)

    # Return the non-empty list
    if len(detected_left_ears) != 0:
        return detected_left_ears
    else:
        return detected_right_ears

# YOLOv5 detector
def detectEarYOLOV5(imgName, imgWidth, imgHeight, model):
    detections = []
    results = model(imgName)

    # For each prediction denormalize the values to get them in pixels
    for tensor in results.xywhn:
        for result in tensor:
            columnNorm, rowNorm, widthNorm, heightNorm, _, _ = result.numpy()
            detections.extend([denormalize(columnNorm, rowNorm, widthNorm, heightNorm, imgWidth, imgHeight)])

    return detections

# Calucate Intersection over union
def calculateIOU(annotation, detected2D, threshold):
    validIoU = 0.0
    # If we don't have any predictions
    if len(detected2D) == 0:
        return validIoU
    # Create the ground-truth box
    annotationBox = [annotation[0], annotation[1], annotation[0] + annotation[2], annotation[1] + annotation[3]]
    
    #For each prediction
    for detected1D in detected2D:
        # Create the prediction box
        detectedBox = [detected1D[0], detected1D[1], detected1D[0] + detected1D[2], detected1D[1] + detected1D[3]]
        # determine the (x, y)-coordinates of the intersection rectangle
        x1 = max(annotationBox[0], detectedBox[0])
        y1 = max(annotationBox[1], detectedBox[1])
        x2 = min(annotationBox[2], detectedBox[2])
        y2 = min(annotationBox[3], detectedBox[3])
        # compute the area of intersection rectangle
        intersection = max(0, y2 - y1 + 1) * max(0, x2 - x1 + 1) 
        # compute the area of the prediction and ground-truth rectangles
        annotationArea = (annotationBox[2] - annotationBox[0] + 1) * (annotationBox[3] - annotationBox[1] + 1)
        detectedArea = (detectedBox[2] - detectedBox[0] + 1) * (detectedBox[3] - detectedBox[1] + 1)
        # compute the union
        union = float(annotationArea + detectedArea - intersection)
        # Compute the IoU
        iou = intersection / union
        # Check if it is valid (bigger or equal the threshold and the biggest)
        if iou >= threshold and iou > validIoU:
            validIoU = iou

    return validIoU

# Calculate simple accuracy
def calculateAccuracy(annotation, detected2D):
    maxAccuracy = 0.0
    # If we don't have any predictions
    if len(detected2D) == 0:
        return maxAccuracy

    # For each prediction
    for detected1D in detected2D:
        # Calculate the accuracy score with the ground-truth
        accuracy = accuracy_score(annotation, detected1D)
        # Return the max accuracy (valid prediction)
        if accuracy > maxAccuracy:
            maxAccuracy = accuracy
    
    return maxAccuracy

# Denormalization (From normalized to pixels)
def denormalize(columnNorm, rowNorm, widthNorm, heightNorm, imgWidth, imgHeight):
    width = round(widthNorm * imgWidth)
    height = round(heightNorm * imgHeight)
    column = round(columnNorm * imgWidth - width / 2)
    row = round(rowNorm * imgHeight - height / 2)

    return [column, row, width, height]