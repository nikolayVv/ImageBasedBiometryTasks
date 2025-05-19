from cv2 import imread

from fnc.segment import segmentation
from fnc.normalize import normalization
from fnc.encode import encoding

# Extract features from an iris image
def featureExtraction(imageFile, eyelashesThresh=80, normalizationRadial=20, normalizationAngular=240, minWaveLength=18, sigmaOnf=0.5):
	# IRIS IMAGE ACQUISITION
	image = imread(imageFile, 0)

	# SEGMENTATION
	coordIris, coordPupil, imageWithNoise = segmentation(image, eyelashesThresh)

	# NORMALIZATION
	polarArray, noiseArray = normalization(imageWithNoise, coordIris[1], coordIris[0], coordIris[2],
										 coordPupil[1], coordPupil[0], coordPupil[2],
										 normalizationRadial, normalizationAngular)

	# FEATURE ENCODING
	template, mask = encoding(polarArray, noiseArray, minWaveLength, sigmaOnf)

	return template, mask, imageFile