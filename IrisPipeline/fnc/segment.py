import numpy as np
from fnc.boundary import searchInnerBound, searchOuterBound
from fnc.line import findline, linecoords
import multiprocessing as mp

# Segment the iris region from the eye image.
# Indicate the noise region.
def segmentation(eyeImage, eyelashesThresh=80):
    # Find the iris boundary by Daugman's intefro-differential
    rowp, colp, rp = searchInnerBound(eyeImage)
    row, col, r = searchOuterBound(eyeImage, rowp, colp, rp)

    # Package pupil and iris boundaries
    rowp = np.round(rowp).astype(int)
    colp = np.round(colp).astype(int)
    rp = np.round(rp).astype(int)
    row = np.round(row).astype(int)
    col = np.round(col).astype(int)
    r = np.round(r).astype(int)
    cirpupil = [rowp, colp, rp]
    ciriris = [row, col, r]

    # Find top and bottom eyelid
    imsz = eyeImage.shape
    irl = np.round(row - r).astype(int)
    iru = np.round(row + r).astype(int)
    icl = np.round(col - r).astype(int)
    icu = np.round(col + r).astype(int)
    if irl < 0:
        irl = 0
    if icl < 0:
        icl = 0
    if iru >= imsz[0]:
        iru = imsz[0] - 1
    if icu >= imsz[1]:
        icu = imsz[1] - 1
    imageIris = eyeImage[irl: iru + 1, icl: icu + 1]

    maskTop = findTopEyelid(imsz, imageIris, irl, icl, rowp, rp)
    maskBot = findBottomEyelid(imsz, imageIris, rowp, rp, irl, icl)

    # Mask the eye image, noise region is masked by NaN value
    imwithnoise = eyeImage.astype(float)
    imwithnoise = imwithnoise + maskTop + maskBot

    ref = eyeImage < eyelashesThresh
    coords = np.where(ref == 1)
    imwithnoise[coords] = np.nan

    return ciriris, cirpupil, imwithnoise


# ------------------------------------------------------------------------------
def findTopEyelid(imageSize, imageIris, irl, icl, rowp, rp, retTop=None):
    topeyelid = imageIris[0: rowp - irl - rp, :]
    lines = findline(topeyelid)
    mask = np.zeros(imageSize, dtype=float)

    if lines.size > 0:
        xl, yl = linecoords(lines, topeyelid.shape)
        yl = np.round(yl + irl - 1).astype(int)
        xl = np.round(xl + icl - 1).astype(int)

        yla = np.max(yl)
        y2 = np.arange(yla)

        mask[yl, xl] = np.nan
        grid = np.meshgrid(y2, xl)
        mask[tuple(grid)] = np.nan

    # Return
    if retTop is not None:
        retTop[0] = mask
    return mask


# ------------------------------------------------------------------------------
def findBottomEyelid(imageSize, imageIris, rowp, rp, irl, icl, retBot=None):
    bottomeyelid = imageIris[rowp - irl + rp - 1: imageIris.shape[0], :]
    lines = findline(bottomeyelid)
    mask = np.zeros(imageSize, dtype=float)

    if lines.size > 0:
        xl, yl = linecoords(lines, bottomeyelid.shape)
        yl = np.round(yl + rowp + rp - 3).astype(int)
        xl = np.round(xl + icl - 2).astype(int)
        yla = np.min(yl)
        y2 = np.arange(yla-1, imageSize[0])

        mask[yl, xl] = np.nan
        grid = np.meshgrid(y2, xl)
        mask[tuple(grid)] = np.nan

    # Return
    if retBot is not None:
        retBot[0] = mask
    return mask
