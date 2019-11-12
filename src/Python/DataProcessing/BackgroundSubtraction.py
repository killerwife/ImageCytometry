import XMLRead
import cv2
import numpy
import os

class BackgroundSubtraction(object):
    def __init__(self, historyCount):
        self.images = []
        self.historyCount = historyCount

    def addImage(self, image):
        if len(self.images) >= self.historyCount:
            self.images.pop(0)
        self.images.append(image)

    def getBackground(self):
        #background = numpy.float32(numpy.zeros_like(self.images[0]))
        background = numpy.float32(self.images[0])
        for i in range(1, len(self.images)):
            cv2.accumulateWeighted(self.images[i], background, 0.5)
        return background

    def getBackgroundFromAll(self, images):
        background = numpy.float32(numpy.zeros_like(images[0]))
        imageCount = len(images)
        for i in range(0, imageCount):
            cv2.accumulateWeighted(images[i], background, 0.01)
        return background

FIRST_VIDEO = True
SINGLE_IMAGE = False
PATH_TO_ANNOTATIONS = 'C:\\GitHubCode\\phd\\ImageCytometry\\src\\XML\\'
if FIRST_VIDEO:
    PATH_TO_IMAGE_ROOT_DIR = 'D:\\BigData\\cellinfluid\\bunkyObrazkyTiff\\'
    ANNOTATIONS_FILE_NAME = 'tracks_1_300.xml'
else:
    PATH_TO_IMAGE_ROOT_DIR = 'D:\\BigData\\cellinfluid\\\deformabilityObrazky\\'
    ANNOTATIONS_FILE_NAME = 'deformabilityAnnotations.xml'

PATH_TO_OUTPUT_ROOT_DIR = PATH_TO_IMAGE_ROOT_DIR
if SINGLE_IMAGE:
    PATH_TO_OUTPUT_ROOT_DIR += 'subtractedBackgroundsSingleImage\\'
else:
    PATH_TO_OUTPUT_ROOT_DIR += 'subtractedBackgrounds\\'

fileNamePredicted = PATH_TO_ANNOTATIONS + ANNOTATIONS_FILE_NAME
annotatedData = []
XMLRead.readXML(fileNamePredicted, annotatedData)

def background_subtraction(gray_image, background_image):
    gray_image_16 = gray_image.astype(numpy.int16)
    background_image_16 = background_image.astype(numpy.int16)
    gray_image_16 = gray_image_16 - background_image_16
    cv2.normalize(gray_image_16, gray_image_16, 0, 255, cv2.NORM_MINMAX)
    gray_image = gray_image_16.astype(numpy.uint8)
    return gray_image

def subtractBackgrounds(annotatedData):
    # fgbg = cv2.bgsegm.createBackgroundSubtractorCNT()
    if SINGLE_IMAGE == False: # history
        bgOwn = BackgroundSubtraction(10)
        for imageData in annotatedData:
            image = cv2.imread(PATH_TO_IMAGE_ROOT_DIR + imageData.filename)
            # output = fgbg.apply(image)
            bgOwn.addImage(image)
            output = background_subtraction(image, bgOwn.getBackground())
            cv2.imwrite(PATH_TO_OUTPUT_ROOT_DIR + imageData.filename, output)
    else: # one background
        bgOwn = BackgroundSubtraction(300)

        images = []
        for imageData in annotatedData:
            image = cv2.imread(PATH_TO_IMAGE_ROOT_DIR + imageData.filename)
            images.append(image)

        i = 0
        background = bgOwn.getBackgroundFromAll(images)
        cv2.imwrite('Derp.png', background)
        for imageData in annotatedData:
            output = background_subtraction(images[i], background)
            cv2.imwrite(PATH_TO_OUTPUT_ROOT_DIR + imageData.filename, output)
            i += 1



def makeDirs(path):
    newDir = path + '1-50'
    if not os.path.exists(newDir):
        os.makedirs(newDir)
    newDir = path + '51-100'
    if not os.path.exists(newDir):
        os.makedirs(newDir)
    if FIRST_VIDEO == False:
        return
    newDir = path + '101-150'
    if not os.path.exists(newDir):
        os.makedirs(newDir)
    newDir = path + '151-200'
    if not os.path.exists(newDir):
        os.makedirs(newDir)
    newDir = path + '201-250'
    if not os.path.exists(newDir):
        os.makedirs(newDir)
    newDir = path + '251-300'
    if not os.path.exists(newDir):
        os.makedirs(newDir)

makeDirs(PATH_TO_OUTPUT_ROOT_DIR)
subtractBackgrounds(annotatedData)
