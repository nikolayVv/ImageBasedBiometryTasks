import numpy as np


# Normalize iris region by unwraping the circular region into a rectangular
# block of constant dimensions.
def normalization(image, xIris, yIris, rIris, xPupil, yPupil, rPupil, radPixels, angularDiv):
	radiusPixels = radPixels + 2
	angleDivisions = angularDiv-1

	r = np.arange(radiusPixels)
	theta = np.linspace(0, 2*np.pi, angleDivisions+1)

	# Calculate displacement of pupil center from the iris center
	ox = xPupil - xIris
	oy = yPupil - yIris

	if ox <= 0:
		sgn = -1
	elif ox > 0:
		sgn = 1

	if ox==0 and oy > 0:
		sgn = 1

	a = np.ones(angleDivisions+1) * (ox**2 + oy**2)

	if ox == 0:
		phi = np.pi/2
	else:
		phi = np.arctan(oy/ox)

	b = sgn * np.cos(np.pi - phi - theta)

	# Calculate radius around the iris as a function of the angle
	r = np.sqrt(a)*b + np.sqrt(a*b**2 - (a - rIris**2))
	r = np.array([r - rPupil])

	rmat = np.dot(np.ones([radiusPixels,1]), r)

	rmat = rmat * np.dot(np.ones([angleDivisions+1,1]),
							np.array([np.linspace(0,1,radiusPixels)])).transpose()
	rmat = rmat + rPupil

	# Exclude values at the boundary of the pupil iris border, and the iris scelra border
	# as these may not correspond to areas in the iris region and will introduce noise.
	rmat = rmat[1 : radiusPixels-1, :]

	# Calculate cartesian location of each data point around the circular iris region
	xcosmat = np.dot(np.ones([radiusPixels-2,1]), np.array([np.cos(theta)]))
	xsinmat = np.dot(np.ones([radiusPixels-2,1]), np.array([np.sin(theta)]))

	xo = rmat * xcosmat
	yo = rmat * xsinmat

	xo = xPupil + xo
	xo = np.round(xo).astype(int)
	coords = np.where(xo >= image.shape[1])
	xo[coords] = image.shape[1] - 1
	coords = np.where(xo < 0)
	xo[coords] = 0
	
	yo = yPupil - yo
	yo = np.round(yo).astype(int)
	coords = np.where(yo >= image.shape[0])
	yo[coords] = image.shape[0] - 1
	coords = np.where(yo < 0)
	yo[coords] = 0

	# Extract intensity values into the normalised polar representation through
	# interpolation
	polarArray = image[yo, xo]
	polarArray = polarArray / 255

	# Create noise array with location of NaNs in polarArray
	polarNoise = np.zeros(polarArray.shape)
	coords = np.where(np.isnan(polarArray))
	polarNoise[coords] = 1

	# Get rid of outling points in order to write out the circular pattern
	image[yo, xo] = 255

	# Get pixel coords for circle around iris
	x,y = circlecoords([xIris,yIris], rIris, image.shape)
	image[y,x] = 255

	# Get pixel coords for circle around pupil
	xp,yp = circlecoords([xPupil,yPupil], rPupil, image.shape)
	image[yp,xp] = 255

	# Replace NaNs before performing feature encoding
	coords = np.where((np.isnan(polarArray)))
	polar_array2 = polarArray
	polar_array2[coords] = 0.5
	avg = np.sum(polar_array2) / (polarArray.shape[0] * polarArray.shape[1])
	polarArray[coords] = avg

	return polarArray, polarNoise.astype(bool)


#------------------------------------------------------------------------------
def circlecoords(center, radius, imgsize, nsides=600):
	a = np.linspace(0, 2*np.pi, 2*nsides+1)
	xd = np.round(radius * np.cos(a) + center[0])
	yd = np.round(radius * np.sin(a) + center[1])

	#  Get rid of values larger than image
	xd2 = xd
	coords = np.where(xd >= imgsize[1])
	xd2[coords[0]] = imgsize[1] - 1
	coords = np.where(xd < 0)
	xd2[coords[0]] = 0

	yd2 = yd
	coords = np.where(yd >= imgsize[0])
	yd2[coords[0]] = imgsize[0] - 1
	coords = np.where(yd < 0)
	yd2[coords[0]] = 0

	x = np.round(xd2).astype(int)
	y = np.round(yd2).astype(int)
	return x,y