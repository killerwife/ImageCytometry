import XMLRead

maxPosDiff = 0.3

def evaluateBoundingBox(original, predicted, overlap=0.5):
    overlapWidth = (original.x + original.width) - predicted.x
    overlapHeight = (original.y + original.height) - predicted.y
    originalArea = original.width * original.height
    distance = (original.x - predicted.x)**2 + (original.y - predicted.y)**2
    if distance < (predicted.width * maxPosDiff)**2 and overlapWidth > 0 and overlapHeight > 0:
        return originalArea * overlap < overlapWidth * overlapHeight
    else:
        return False

def evaluateImage(originalBoxes, predictedBoxes):
    hits = 0
    missed = 0
    falseAlarm = 0
    totalBoxes = len(originalBoxes)
    foundRefs = [False] * totalBoxes
    for predictedBox in predictedBoxes:
        found = False
        i = 0
        while i < len(originalBoxes):
            success = evaluateBoundingBox(originalBoxes[i], predictedBox)
            if success:
                found = True
                foundRefs[i] = True
            i += 1
        if found == False:
            falseAlarm = falseAlarm + 1
    i = 0
    while i < len(foundRefs):
        if foundRefs[i]:
            hits = hits + 1
        else:
            missed = missed + 1
        i += 1
    return hits, missed, falseAlarm, totalBoxes

def processImages(annotatedData, predictedData, barrier):
    totalHits = 0
    totalMissed = 0
    totalFalseAlarms = 0
    totalObjects = 0
    i = 0
    while i < len(annotatedData):
        if i < barrier:
            i += 1
            continue
        hits, missed, falseAlarm, totalBoxes = evaluateImage(annotatedData[i].boundingBoxes, predictedData[i].boundingBoxes)
        totalHits += hits
        totalMissed += missed
        totalFalseAlarms += falseAlarm
        totalObjects += totalBoxes
        print('File{:s} Hits:{:d} Missed:{:d} FalseAlarms:{:d} Objects:{:d}'.format(annotatedData[i].filename, hits, missed, falseAlarm, totalBoxes))
        i += 1
    print('\nTotal: Hits:{:d} Missed:{:d} FalseAlarms:{:d} Objects:{:d} Precision:{:.1f}% Recall:{:.1f}%'.format(
        totalHits, totalMissed, totalFalseAlarms, totalObjects, float(totalHits)/(totalHits + totalFalseAlarms) * 100, float(totalHits)/(totalHits + totalMissed) * 100))


FIRST_VIDEO = False
MODEL_NAME = 'model08042019-250And50'
if FIRST_VIDEO == True:
    ANNOTATIONS_FILE_NAME = 'tracks_1_300.xml'
    BARRIER = 250
else:
    ANNOTATIONS_FILE_NAME = 'deformabilityAnnotations.xml'
    BARRIER = 50
    MODEL_NAME += 'SecondVideo'
fileNamePredicted = 'C:\\GitHubCode\\phd\\ImageCytometry\\src\\XML\\' + ANNOTATIONS_FILE_NAME
fileNameAnnotated = 'D:\\BigData\\cellinfluid\\Annotations\\PredictedAnnotations\\tracks_1_300_' + MODEL_NAME + '.xml'
annotatedData = []
predictedData = []
XMLRead.readXML(fileNamePredicted, annotatedData)
XMLRead.readXML(fileNameAnnotated, predictedData)
processImages(annotatedData, predictedData, BARRIER)






