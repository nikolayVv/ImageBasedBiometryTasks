import cv2
from images import loadAndConvertImages2D, convertTo1D, convertToScikitLBP1D, convertToLBP1D
from classification import calculateRanks

if __name__ == '__main__':
    dimSize = (64, 64)
    #dimSize = (128, 128)
    radius, codeLength = 1, 8
    #radius, codeLength = 2, 16

    print("Doing classification for 1000 photos size: " + str(dimSize[0]) + "x" + str(dimSize[1]))
    print("Using LBP with radius: " + str(radius) + ", and code length: " + str(codeLength))

    print()
    print("Loading samples...")
    samples2D = loadAndConvertImages2D('awe', dimSize, cv2.COLOR_BGR2GRAY)
    print("Samples loaded.")
    print("Converting to Histogram and 1D array...")
    samples1D, samples1DHist = convertTo1D(samples2D)
    print("Converting to histogram and 1D array finished.")

    print("Converting to LBP and LBP Histogram...")
    samples1DLBP, samples1DLBPHist = convertToLBP1D(samples2D, dimSize[0], dimSize[1], radius, codeLength, "default")
    print("Converting to LBP and LBP Histogram finished.")
    print("Converting to LBP Uniform and LBP Uniform Histogram...")
    samples1DLBPUniform, samples1DLBPUniformHist = convertToLBP1D(samples2D, dimSize[0], dimSize[1], radius, codeLength, "uniform")
    print("Converting to LBP Uniform and LBP Uniform Histogram finished.")


    print("Converting to LBP Scikit and LBP Scikit Histogram...")
    samples1DLBPScikit, samples1DLBPScikitHist = convertToScikitLBP1D(samples2D, radius, codeLength, "default")
    print("Converting to LBP Scikit and LBP Scikit Histogram finished.")
    print("Converting to LBP Scikit Uniform and LBP Scikit Uniform Histogram...")
    samples1DLBPScikitUniform, samples1DLBPScikitUniformHist = convertToScikitLBP1D(samples2D, radius, codeLength, "uniform")
    print("Converting to LBP Scikit Uniform and LBP Scikit Uniform Histogram finished.")

    print()
    print("Starting classification...")
    rank1Euclidian, rank1Manhattan, rank1Cosines = calculateRanks(samples1D)
    print("Pixel by pixel classification done.")
    rank1EuclidianHist, rank1ManhattanHist, rank1CosinesHist = calculateRanks(samples1DHist)
    print("Pixel by pixel histogram classification done.")

    rank1EuclidianLBP, rank1ManhattanLBP, rank1CosinesLBP = calculateRanks(samples1DLBP)
    print("LBP classification done.")
    rank1EuclidianLBPHist, rank1ManhattanLBPHist, rank1CosinesLBPHist = calculateRanks(samples1DLBPHist)
    print("LBP histogram classification done.")
    rank1EuclidianLBPUniform, rank1ManhattanLBPUniform, rank1CosinesLBPUniform = calculateRanks(samples1DLBPUniform)
    print("LBP uniform classification done.")
    rank1EuclidianLBPUniformHist, rank1ManhattanLBPUniformHist, rank1CosinesLBPUniformHist = calculateRanks(samples1DLBPUniformHist)
    print("LBP uniform histogram classification done.")

    rank1EuclidianLBPScikit, rank1ManhattanLBPScikit, rank1CosinesLBPScikit = calculateRanks(samples1DLBPScikit)
    print("LBP Scikit classification done.")
    rank1EuclidianLBPScikitHist, rank1ManhattanLBPScikitHist, rank1CosinesLBPScikitHist = calculateRanks(samples1DLBPScikitHist)
    print("LBP Scikit histogram classification done.")
    rank1EuclidianLBPScikitUniform, rank1ManhattanLBPScikitUniform, rank1CosinesLBPScikitUniform = calculateRanks(samples1DLBPScikitUniform)
    print("LBP Scikit uniform classification done.")
    rank1EuclidianLBPScikitUniformHist, rank1ManhattanLBPScikitUniformHist, rank1CosinesLBPScikitUniformHist = calculateRanks(samples1DLBPScikitUniformHist)
    print("LBP Scikit uniform histogram classification done.")


    print()
    print("---RESULTS---")
    print("--PIXEL BY PIXEL--")
    print("Rank-1 of Euclidian method: ", rank1Euclidian)
    print("Rank-1 of Manhattan method: ", rank1Manhattan)
    print("Rank-1 of Cosines method: ", rank1Cosines)
    print("--PIXEL BY PIXEL HISTOGRAM--")
    print("Rank-1 of Euclidian method: ", rank1EuclidianHist)
    print("Rank-1 of Manhattan method: ", rank1ManhattanHist)
    print("Rank-1 of Cosines method: ", rank1CosinesHist)
    print("--LBP--")
    print("Rank-1 of Euclidian method: ", rank1EuclidianLBP)
    print("Rank-1 of Manhattan method: ", rank1ManhattanLBP)
    print("Rank-1 of Cosines method: ", rank1CosinesLBP)
    print("--LBP HISTOGRAM--")
    print("Rank-1 of Euclidian method: ", rank1EuclidianLBPHist)
    print("Rank-1 of Manhattan method: ", rank1ManhattanLBPHist)
    print("Rank-1 of Cosines method: ", rank1CosinesLBPHist)
    print("--LBP UNIFORM--")
    print("Rank-1 of Euclidian method: ", rank1EuclidianLBPUniform)
    print("Rank-1 of Manhattan method: ", rank1ManhattanLBPUniform)
    print("Rank-1 of Cosines method: ", rank1CosinesLBPUniform)
    print("--LBP UNIFORM HISTOGRAM--")
    print("Rank-1 of Euclidian method: ", rank1EuclidianLBPUniformHist)
    print("Rank-1 of Manhattan method: ", rank1ManhattanLBPUniformHist)
    print("Rank-1 of Cosines method: ", rank1CosinesLBPUniformHist)
    print("--LBP SCIKIT--")
    print("Rank-1 of Euclidian method: ", rank1EuclidianLBPScikit)
    print("Rank-1 of Manhattan method: ", rank1ManhattanLBPScikit)
    print("Rank-1 of Cosines method: ", rank1CosinesLBPScikit)
    print("--LBP SCIKIT HISTOGRAM--")
    print("Rank-1 of Euclidian method: ", rank1EuclidianLBPScikitHist)
    print("Rank-1 of Manhattan method: ", rank1ManhattanLBPScikitHist)
    print("Rank-1 of Cosines method: ", rank1CosinesLBPScikitHist)
    print("--LBP UNIFORM SCIKIT--")
    print("Rank-1 of Euclidian method: ", rank1EuclidianLBPScikitUniform)
    print("Rank-1 of Manhattan method: ", rank1ManhattanLBPScikitUniform)
    print("Rank-1 of Cosines method: ", rank1CosinesLBPScikitUniform)
    print("--LBP UNIFORM SCIKIT HISTOGRAM--")
    print("Rank-1 of Euclidian method: ", rank1EuclidianLBPScikitUniformHist)
    print("Rank-1 of Manhattan method: ", rank1ManhattanLBPScikitUniformHist)
    print("Rank-1 of Cosines method: ", rank1CosinesLBPScikitUniformHist)
    print("---END---")