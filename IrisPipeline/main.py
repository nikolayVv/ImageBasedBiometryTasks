from fnc.extractFeature import featureExtraction
from fnc.matching import calculateHammingDistance
from scipy.io import savemat, loadmat
import os
import glob

DIR = "./data"
TEMP_DIR = "./templates"
ENROLLMENT_FILES = "1"
COMPARISON_FILES = "2"
ENROLLMENT_ENABLED = True
EVALUATION_ENABLED = True
THRESHOLD = 0.5
RANK = 1


if __name__ == '__main__':
    # ENROLLMENT PART
    if ENROLLMENT_ENABLED == True:
        if not os.path.isdir(TEMP_DIR):
            os.mkdir(TEMP_DIR)

        for className in os.listdir(DIR):
            if os.path.isdir(DIR + '/' + className):
                imagesSegmAndNorm = []
                print("Enrolling class " + className + "...")
                for imageFile in glob.glob(DIR + "/" + className + "/" + ENROLLMENT_FILES + "/*_*_*.jpg"):
                    # FEATURE EXTRACTION
                    template, mask, _ = featureExtraction(imageFile)
                    
                    # ENROLLMENT
                    templateName = os.path.basename(imageFile)
                    out_file = os.path.join(TEMP_DIR, "%s.mat" % (templateName.split(".")[0]))
                    savemat(out_file, mdict={'template': template, 'mask': mask})
                    print(imageFile.split("\\")[1].split(".")[0], ".mat")
    
    # SEGMENTATION PART
    if EVALUATION_ENABLED == True:
        truePositives = 0
        trueNegatives = 0
        falsePositives = 0
        falseNegatives = 0

        for className in os.listdir(DIR):
            if os.path.isdir(DIR + '/' + className):
                print("Evaluating class " + className + "...")
                for imageFile in glob.glob(DIR + "/" + className + "/" + COMPARISON_FILES + "/*_*_*.jpg"):
                    
                    # FEATURE EXTRACTION
                    template, mask, _ = featureExtraction(imageFile)
                    
                    # COMPARISON
                    minDistanceFile = ''
                    minDistance = -1
                    if RANK != 1:
                        isInRank = False
                        rank = RANK
                        
                    classExistsAsTemplate = False
                    for matFile in glob.glob(TEMP_DIR + "/*.mat"):
                        mat = loadmat(matFile)
                        templateDB = mat['template']
                        maskDB = mat['mask']

                        currDistance = calculateHammingDistance(template, mask, templateDB, maskDB)
                        
                        if minDistance <= THRESHOLD:
                                if (minDistance == -1 and minDistanceFile == '') or minDistance > currDistance:
                                    minDistance = currDistance
                                    minDistanceFile = matFile
                                    if RANK != 1:
                                        if isInRank == True:
                                            rank -= 1
                                        if rank == 0:
                                            isInRank = False
                                        if matFile.split("\\")[1].split(".")[0].split("_")[0] == className:
                                            isInRank = True
                                            rank = RANK


                        if classExistsAsTemplate == False and matFile.split("\\")[1].split(".")[0].split("_")[0] == className:
                            classExistsAsTemplate = True
                    
                    minDistanceClass = minDistanceFile.split("\\")[1].split(".")[0].split("_")[0]

                    # MATCH SCORES
                    if (RANK != 1 and isInRank == True) or minDistanceClass == className:
                        print("True positive => ", minDistanceClass)
                        truePositives += 1
                    elif ((RANK != 1 and isInRank == True) or (minDistance == -1 and minDistanceFile == '')) and classExistsAsTemplate == False:
                        print("True negative => ", minDistanceClass)
                        trueNegatives += 1
                    elif ((RANK != 1 and isInRank == False) or minDistanceClass != className) and classExistsAsTemplate == True:
                        print("False positive => ", minDistanceClass)
                        falsePositives += 1
                    elif ((RANK != 1 and isInRank == False) or minDistanceClass != className) and classExistsAsTemplate == False:
                        print("False negative => ", minDistanceClass)
                        falseNegatives += 1

        # TP + TN / TP + TN + FP + FN
        accuracy = (truePositives + trueNegatives) / (truePositives + trueNegatives + falsePositives + falseNegatives)
        # TP / TP+FP
        precision = truePositives / (truePositives + falsePositives)
        # TP / TP+FN
        recall = truePositives / (truePositives + falseNegatives)
        f1score = 2*precision*recall / (precision+recall)

        print("-----RESULTS-----")
        print("Accuracy: ", round(accuracy*100,2), "%")
        print("Precision: ", round(precision*100,2), "%")
        print("Recall: ", round(recall*100,2), "%")
        print("F1-score: ", round(f1score*100,2), "%")
        print("-----END-----")
