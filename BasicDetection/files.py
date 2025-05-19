import cv2
from detector import denormalize

# Load images
def loadImage(imageName):
    image = cv2.imread(imageName)
    imageHeight = image.shape[0]
    imageWidth = image.shape[1]        
    # Make the image gray
    imageGray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    return imageGray, imageWidth, imageHeight

# Load ground-truth
def loadAnnotation(imageName, imageWidth, imageHeight):
    annotationName = imageName.split(".")[0] + ".txt"

    with open(annotationName) as annotation:
        lines = annotation.readlines()
        annotationNorm = []

        for line in lines:
            for val in line.split(" ")[1:]:
                annotationNorm.append(float(val))
    
    return denormalize(annotationNorm[0], annotationNorm[1], annotationNorm[2], annotationNorm[3], imageWidth, imageHeight)