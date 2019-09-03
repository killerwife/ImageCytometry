import XMLRead
import cv2

PATH_TO_XML_ROOT_DIR = 'D:\\BigData\\cellinfluid\\bunkyObrazkyTiff\\'
PATH_TO_OUTPUT_ROOT_DIR = 'D:\\BigData\\cellinfluid\\subtractedBackgrounds\\'

fileNamePredicted = PATH_TO_XML_ROOT_DIR + 'tracks_1_200.xml'
annotatedData = []
XMLRead.readXML(fileNamePredicted, annotatedData)

def subtractBackgrounds(annotatedData):
    fgbg = cv2.bgsegm.createBackgroundSubtractorMOG(backgroundRatio=0.5)
    for imageData in annotatedData:
        image = cv2.imread(PATH_TO_XML_ROOT_DIR + imageData.filename)
        output = fgbg.apply(image)
        cv2.imwrite(PATH_TO_OUTPUT_ROOT_DIR + imageData.filename, output)


subtractBackgrounds(annotatedData)
