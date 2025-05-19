from scipy.spatial import distance
from images import haveSameClass
import math

# Calculates the Euclidian, Manhattan and Cosines distances between all the images from all the classes
# Finds the closest and checks if they are in the same class
# The function returns Rank-1 for each of the distances by deviding the match score with all possible images (1000)
def calculateRanks(images):
    matchEuc, matchMan, matchCos = 0, 0, 0

    for i in range(len(images)):
        minEuc, minMan, minCos = None, None, None
        indexEuc, indexMan, indexCos = None, None, None

        print("Calculating distances from image " + str((i % 10) + 1) + " from class " + str(math.floor(i / 10) + 1) + " to all other images.")
        for j in range(len(images)):
            if i == j:
                continue
            
            newEuc = distance.euclidean(images[i], images[j])
            newMan = distance.cityblock(images[i], images[j])
            newCos = distance.cosine(images[i], images[j])

            if minEuc == None or newEuc < minEuc:
                minEuc = newEuc
                indexEuc = j
            if minMan == None or newMan < minMan:
                minMan = newMan
                indexMan = j
            if minCos == None or newCos < minCos:
                minCos = newCos
                indexCos = j
        if indexEuc != None and haveSameClass(i, indexEuc):
            matchEuc += 1
        if indexMan != None and haveSameClass(i, indexMan):
            matchMan += 1
        if indexCos != None and haveSameClass(i, indexCos):
            matchCos += 1
    
    return matchEuc/1000, matchMan/1000, matchCos/1000
