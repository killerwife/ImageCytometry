import XMLRead
import cv2
import math
import os

class DatasetSegment(object):
    def __init__(self, startExclusion, endExclusion, startBoundary, endBoundary):
        self.startExclusion = startExclusion
        self.endExclusion = endExclusion
        self.startBoundary = startBoundary
        self.endBoundary = endBoundary

maxPosDiff = float(0.5)
PATH_TO_IMAGE_ROOT_DIR = 'D:\\BigData\\cellinfluid\\bunkyObrazkyTiff\\'

def evaluateBoundingBox(original, predicted):
    biggerLeft = 0
    biggerRight = 0
    if original.x > predicted.x:
        biggerLeft = original.x
    else:
        biggerLeft = predicted.x
    if original.x + original.width < predicted.x + predicted.width:
        biggerRight = original.x + original.width
    else:
        biggerRight = predicted.x + predicted.width
    overlapWidth = biggerRight - biggerLeft
    if original.y > predicted.y:
        biggerLeft = original.y
    else:
        biggerLeft = predicted.y
    if original.y + original.height < predicted.y + predicted.height:
        biggerRight = original.y + original.height
    else:
        biggerRight = predicted.y + predicted.height
    overlapHeight = biggerRight - biggerLeft
    originalArea = original.width * original.height
    distance = ((original.x + original.width / 2) - (predicted.x + predicted.width / 2))**2 + ((original.y + original.height / 2) - (predicted.y + predicted.height / 2))**2
    distance = math.sqrt(float(distance))
    overlapMaxdist = predicted.width * maxPosDiff
    if distance < overlapMaxdist and overlapWidth > 0 and overlapHeight > 0:
        return float(overlapWidth) * overlapHeight / originalArea, distance / predicted.width
    else:
        return float(0), float(0)

def evaluateImage(originalBoxes, predictedBoxes):
    hits = []
    hits.append(0)
    hits.append(0)
    hits.append(0)
    distances = []
    distances.append(0)
    distances.append(0)
    missed = 0
    falseAlarm = 0
    totalBoxes = len(originalBoxes)
    foundRefs = [False] * totalBoxes
    falsePositives = []
    falseNegatives = []
    for predictedBox in predictedBoxes:
        i = 0
        highestOverlap = 0
        highestIndex = -1
        highestDistRatio = 0
        if predictedBox.x == 406 and predictedBox.y == 591:
            print()
        while i < len(originalBoxes):
            overlapRatio, centerDistRatio = evaluateBoundingBox(originalBoxes[i], predictedBox)
            if overlapRatio > highestOverlap:
                highestOverlap = overlapRatio
                highestIndex = i
                highestDistRatio = centerDistRatio

            i += 1
        if highestOverlap < 0.1:
            falseAlarm = falseAlarm + 1
            falsePositives.append(predictedBox)
        else:
            foundRefs[highestIndex] = True
            if highestOverlap < 0.3:
                hits[2] += 1
            elif highestOverlap < 0.5:
                hits[1] += 1
            else:
                hits[0] += 1
            if highestDistRatio < 0.3:
                distances[0] += 1
            else:
                distances[1] += 1
    i = 0
    while i < len(foundRefs):
        if foundRefs[i] == False:
            missed = missed + 1
            falseNegatives.append(originalBoxes[i])
        i += 1
    return hits, missed, falseAlarm, totalBoxes, falsePositives, falseNegatives, distances

def processImages(annotatedData, predictedData, datasetSegment, modelName, partialOutputs):
    totalHits = []
    totalHits.append(0)
    totalHits.append(0)
    totalHits.append(0)
    totalDistances = []
    totalDistances.append(0)
    totalDistances.append(0)
    totalMissed = 0
    totalFalsePositives = 0
    totalObjects = 0
    i = 0
    while i < len(annotatedData):
        if i < datasetSegment.startExclusion:
            i += 1
            continue
        hits, missed, falseAlarm, totalBoxes, falsePositives, falseNegatives, distances = evaluateImage(annotatedData[i].boundingBoxes, predictedData[i].boundingBoxes)
        totalHits[0] += hits[0]
        totalHits[1] += hits[1]
        totalHits[2] += hits[2]
        totalDistances[0] += distances[0]
        totalDistances[1] += distances[1]
        totalMissed += missed
        totalFalsePositives += falseAlarm
        totalObjects += totalBoxes
        if partialOutputs:
            print('File{:s} Hits:0.5>{:d} 0.3>{:d} 0.1>{:d} Missed:{:d} FalseAlarms:{:d} Objects:{:d}'.format(annotatedData[i].filename, hits[0], hits[1], hits[2], missed, falseAlarm, totalBoxes))
        # image = cv2.imread(PATH_TO_IMAGE_ROOT_DIR + annotatedData[i].filename)
        # k = 0
        # for predictedBox in falsePositives:
        #     cutout = image[predictedBox.y:predictedBox.y+predictedBox.height, predictedBox.x:predictedBox.x+predictedBox.width]
        #     cv2.imwrite('FalsePos\\' + str(i) + 'image' + str(k) + '.png', cutout)
        #     k += 1
        #     # cv2.rectangle(image, (predictedBox.x, predictedBox.y), (predictedBox.x + predictedBox.width, predictedBox.y + predictedBox.height), (0, 255, 0), 3)
        # k = 0
        # for predictedBox in falseNegatives:
        #     cutout = image[predictedBox.y:predictedBox.y+predictedBox.height, predictedBox.x:predictedBox.x+predictedBox.width]
        #     cv2.imwrite('FalseNeg\\' + str(i) + 'image' + str(k) + '.png', cutout)
        #     k += 1
            # cv2.rectangle(image, (predictedBox.x, predictedBox.y), (predictedBox.x + predictedBox.width, predictedBox.y + predictedBox.height), (0, 0, 255), 3)
        # cv2.imshow("test",image)
        # cv2.waitKey()
        i += 1
        precision = 0
        if totalHits[0] + totalHits[1] + totalHits[2] + totalFalsePositives > 0:
            precision = float(totalHits[0] + totalHits[1] + totalHits[2])/(totalHits[0] + totalHits[1] + totalHits[2] + totalFalsePositives) * 100
        recall = 0
        if totalHits[0] + totalHits[1] + totalHits[2] + totalMissed > 0:
            recall =  float(totalHits[0] + totalHits[1] + totalHits[2])/(totalHits[0] + totalHits[1] + totalHits[2] + totalMissed) * 100
        if i >= datasetSegment.endExclusion:
            break
    print('\n' + modelName + ' - Total: Hits: 0.5>{:d} 0.3>{:d} 0.1>{:d} Distances: 0.3<{:d} 0.5<{:d} Missed:{:d} FalseAlarms:{:d} Objects:{:d} Precision:{:.1f}% Recall:{:.1f}%'.format(
        totalHits[0], totalHits[1], totalHits[2], totalDistances[0], totalDistances[1], totalMissed, totalFalsePositives, totalObjects, precision, recall))


def loadAndProcess(firstVideo, modelName, datasetSegment):
    if firstVideo:
        ANNOTATIONS_FILE_NAME = 'tracks_1_300.xml'
    else:
        ANNOTATIONS_FILE_NAME = 'deformabilityAnnotations.xml'
        modelName += 'SecondVideo'
    fileNamePredicted = 'C:\\GitHubCode\\phd\\ImageCytometry\\src\\XML\\' + ANNOTATIONS_FILE_NAME
    fileNameAnnotated = 'D:\\BigData\\cellinfluid\\Annotations\\PredictedAnnotations\\tracks_1_300_' + modelName + '.xml'
    annotatedData = []
    predictedData = []
    XMLRead.readXML(fileNamePredicted, annotatedData)
    XMLRead.readXML(fileNameAnnotated, predictedData)
    processImages(annotatedData, predictedData, datasetSegment, modelName, False)

newDir = 'FalseNeg'
if not os.path.exists(newDir):
    os.makedirs(newDir)

newDir = 'FalsePos'
if not os.path.exists(newDir):
    os.makedirs(newDir)

# loadAndProcess(True, 'model21032019-250NoBackground100000')
# loadAndProcess(True, 'model21032019_02-250NoBackground100000')
# loadAndProcess(True, 'model08042019-250NoBackground100000')
#
# loadAndProcess(False, 'model21032019-250NoBackground100000')
# loadAndProcess(False, 'model21032019_02-250NoBackground100000')
# loadAndProcess(False, 'model08042019-250NoBackground100000')
#
# loadAndProcess(True, 'model21032019-200')
# loadAndProcess(True, 'model21032019_02-200')
# loadAndProcess(True, 'model08042019-200')
# loadAndProcess(True, 'fixedSSDExperiment100000')
#
# loadAndProcess(False, 'model21032019-200')
# loadAndProcess(False, 'model21032019_02-200')
# loadAndProcess(False, 'model08042019-200')
# loadAndProcess(False, 'fixedSSDExperiment100000')
#
# loadAndProcess(True, 'model21032019-250')
# loadAndProcess(True, 'model21032019_02-250')
# loadAndProcess(True, 'model08042019-250')
# loadAndProcess(True, 'fixedSSDExperiment250100000')
#
# loadAndProcess(False, 'model21032019-250')
# loadAndProcess(False, 'model21032019_02-250')
# loadAndProcess(False, 'model08042019-250')
# loadAndProcess(False, 'fixedSSDExperiment250100000')

loadAndProcess(True, 'model21032019-250And50', DatasetSegment(250, 300, 200, 250))
loadAndProcess(True, 'model21032019_02-250And50', DatasetSegment(250, 300, 200, 250))
loadAndProcess(True, 'model08042019-250And50', DatasetSegment(250, 300, 200, 250))
loadAndProcess(True, 'fixedSSDExperiment250And50100000', DatasetSegment(250, 300, 200, 250))
loadAndProcess(True, 'fixedSSDExperimentV2250And50100000', DatasetSegment(250, 300, 200, 250))
loadAndProcess(True, 'fixedSSDExperimentV3250And50100000', DatasetSegment(250, 300, 200, 250))

loadAndProcess(False, 'model21032019-250And50', DatasetSegment(50, 100, 30, 50))
loadAndProcess(False, 'model21032019_02-250And50', DatasetSegment(50, 100, 30, 50))
loadAndProcess(False, 'model08042019-250And50', DatasetSegment(50, 100, 30, 50))
loadAndProcess(False, 'fixedSSDExperiment250And50100000', DatasetSegment(50, 100, 30, 50))
loadAndProcess(False, 'fixedSSDExperimentV2250And50100000', DatasetSegment(50, 100, 30, 50))
loadAndProcess(False, 'fixedSSDExperimentV3250And50100000', DatasetSegment(50, 100, 30, 50))

loadAndProcess(True, 'model21032019_02-250And50-1100000', DatasetSegment(0, 50, 200, 250))
loadAndProcess(True, 'model21032019_02-250And50-2100000', DatasetSegment(50, 100, 100, 150))
loadAndProcess(True, 'model21032019_02-250And50-3100000', DatasetSegment(100, 150, 200, 250))
loadAndProcess(True, 'model21032019_02-250And50-4100000', DatasetSegment(150, 200, 0, 50))
loadAndProcess(True, 'model21032019-250And50-1100000', DatasetSegment(0, 50, 200, 250))
loadAndProcess(True, 'model21032019-250And50-2100000', DatasetSegment(50, 100, 100, 150))
loadAndProcess(True, 'model21032019-250And50-3100000', DatasetSegment(100, 150, 200, 250))
loadAndProcess(True, 'model21032019-250And50-4100000', DatasetSegment(150, 200, 0, 50))
loadAndProcess(True, 'model08042019-250And50-1100000', DatasetSegment(0, 50, 200, 250))

loadAndProcess(False, 'model21032019_02-250And50-1100000', DatasetSegment(0, 50, 80, 100))
loadAndProcess(False, 'model21032019_02-250And50-2100000', DatasetSegment(0, 50, 50, 70))
loadAndProcess(False, 'model21032019_02-250And50-3100000', DatasetSegment(50, 100, 0, 20))
loadAndProcess(False, 'model21032019_02-250And50-4100000', DatasetSegment(50, 100, 20, 40))
loadAndProcess(False, 'model21032019-250And50-1100000', DatasetSegment(0, 50, 80, 100))
loadAndProcess(False, 'model21032019-250And50-2100000', DatasetSegment(0, 50, 50, 70))
loadAndProcess(False, 'model21032019-250And50-3100000', DatasetSegment(50, 100, 0, 20))
loadAndProcess(False, 'model21032019-250And50-4100000', DatasetSegment(50, 100, 20, 40))
loadAndProcess(False, 'model08042019-250And50-1100000', DatasetSegment(0, 50, 80, 100))






