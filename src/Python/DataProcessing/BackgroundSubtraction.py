import XMLRead
import cv2
import numpy
import os
from enum import Enum

class BackgroundSubtraction(object):
    def __init__(self, historyCount):
        self.images = []
        self.historyCount = historyCount

    def addImage(self, image):
        if len(self.images) >= self.historyCount:
            self.images.pop(0)
        self.images.append(image)

    def getBackground(self):
        background = numpy.float32(numpy.zeros_like(self.images[0]))
        for i in range(0, len(self.images)):
            cv2.accumulateWeighted(self.images[i], background, 0.5)
        return background

    def getBackgroundFromAll(self, images):
        background = numpy.float32(numpy.zeros_like(images[0]))
        imageCount = len(images)
        for i in range(0, imageCount):
            cv2.accumulateWeighted(images[i], background, 0.1)
        return background


class BgMode(Enum):
    HISTORY = 0
    SINGLE_IMAGE = 1
    NO_NORMAL = 2

class Video(Enum):
    CANAL = 0
    DEFORMABILITY = 1
    BETKA = 2
    DALSIE = 3

VIDEO = Video.BETKA
MODE = BgMode.NO_NORMAL
XML = True
PATH_TO_ANNOTATIONS = 'C:\\GitHubCode\\phd\\ImageCytometry\\src\\XML\\'
if VIDEO == Video.CANAL:
    PATH_TO_IMAGE_ROOT_DIR = 'D:\\BigData\\cellinfluid\\bunkyObrazkyTiff\\'
    ANNOTATIONS_FILE_NAME = 'tracks_1_300.xml'
elif VIDEO == Video.DEFORMABILITY:
    PATH_TO_IMAGE_ROOT_DIR = 'D:\\BigData\\cellinfluid\\deformabilityObrazky\\'
    ANNOTATIONS_FILE_NAME = 'deformabilityAnnotationsAll.xml'
elif VIDEO == Video.BETKA:
    PATH_TO_IMAGE_ROOT_DIR = 'D:\\BigData\\cellinfluid\\BetkaVideo\\'
    ANNOTATIONS_FILE_NAME = 'betkaAnnotations.xml'
elif VIDEO == Video.DALSIE:
    XML = False
    PATH_TO_IMAGE_ROOT_DIR = '' # pridat cestu do PATH_TO_IMAGE_ROOT_DIR kde su vsetky obrazky

PATH_TO_OUTPUT_ROOT_DIR = PATH_TO_IMAGE_ROOT_DIR
if MODE == BgMode.SINGLE_IMAGE:
    PATH_TO_OUTPUT_ROOT_DIR += 'subtractedBackgroundsSingleImage\\'
elif MODE == BgMode.HISTORY:
    PATH_TO_OUTPUT_ROOT_DIR += 'subtractedBackgrounds\\'
elif MODE == BgMode.NO_NORMAL:
    PATH_TO_OUTPUT_ROOT_DIR += 'subtractedBackgroundsNoNormal\\'

fileNamePredicted = PATH_TO_ANNOTATIONS + ANNOTATIONS_FILE_NAME
annotatedData = []

if XML:
    XMLRead.readXML(fileNamePredicted, annotatedData)
else:
    for r, d, f in os.walk(PATH_TO_IMAGE_ROOT_DIR):
        for annotatedData in f:
            if '.txt' in annotatedData:
                imageData = XMLRead.Image()
                imageData.filename = os.path.join(r, annotatedData)
                annotatedData.append(imageData)


def background_subtraction(gray_image, background_image):
    gray_image_16 = gray_image.astype(numpy.int16)
    background_image_16 = background_image.astype(numpy.int16)
    gray_image_16 = gray_image_16 - background_image_16
    cv2.normalize(gray_image_16, gray_image_16, 0, 255, cv2.NORM_MINMAX)
    gray_image = gray_image_16.astype(numpy.uint8)
    return gray_image


def background_subtraction_old(gray_image, background_image):
    gray_image_16 = gray_image.astype(numpy.int16)
    background_image_16 = background_image.astype(numpy.int16)
    gray_image_16 = gray_image_16 - background_image_16
    gray_image_16 = numpy.absolute(gray_image_16)
    gray_image = gray_image_16.astype(numpy.uint8)
    # cv2.threshold(gray_image, 0, 512, cv2.THRESH_TOZERO)
    cv2.normalize(gray_image, gray_image, 0, 255, cv2.NORM_MINMAX)
    return gray_image


def subtractBackgrounds(annotatedData):
    # fgbg = cv2.bgsegm.createBackgroundSubtractorCNT()
    if MODE == BgMode.HISTORY: # history
        bgOwn = BackgroundSubtraction(10)
        for imageData in annotatedData:
            image = cv2.imread(PATH_TO_IMAGE_ROOT_DIR + imageData.filename)
            # output = fgbg.apply(image)
            bgOwn.addImage(image)
            output = background_subtraction(image, bgOwn.getBackground())
            cv2.imwrite(PATH_TO_OUTPUT_ROOT_DIR + imageData.filename, output)
    elif MODE == BgMode.SINGLE_IMAGE: # one background
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
    elif MODE == BgMode.NO_NORMAL:
        bgOwn = BackgroundSubtraction(10)
        for imageData in annotatedData:
            image = cv2.imread(PATH_TO_IMAGE_ROOT_DIR + imageData.filename)
            # output = fgbg.apply(image)
            bgOwn.addImage(image)
            output = background_subtraction_old(image, bgOwn.getBackground())
            cv2.imwrite(PATH_TO_OUTPUT_ROOT_DIR + imageData.filename, output)


def makeDirs(path):
    if not os.path.exists(PATH_TO_OUTPUT_ROOT_DIR):
        os.makedirs(PATH_TO_OUTPUT_ROOT_DIR)
    if VIDEO == Video.BETKA:
        newDir = path + 'Frames'
        if not os.path.exists(newDir):
            os.makedirs(newDir)
        return

    newDir = path + '1-50'
    if not os.path.exists(newDir):
        os.makedirs(newDir)
    newDir = path + '51-100'
    if not os.path.exists(newDir):
        os.makedirs(newDir)
    if VIDEO == Video.DEFORMABILITY:
        newDir = path + '101-239'
        if not os.path.exists(newDir):
            os.makedirs(newDir)
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
