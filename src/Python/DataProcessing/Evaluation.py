import XMLRead
import cv2
import math

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
    falseAlarms = []
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
            falseAlarms.append(predictedBox)
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
        i += 1
    return hits, missed, falseAlarm, totalBoxes, falseAlarms, distances

def processImages(annotatedData, predictedData, barrier_low, barrier_high, modelName, partialOutputs):
    totalHits = []
    totalHits.append(0)
    totalHits.append(0)
    totalHits.append(0)
    totalDistances = []
    totalDistances.append(0)
    totalDistances.append(0)
    totalMissed = 0
    totalFalseAlarms = 0
    totalObjects = 0
    i = 0
    while i < len(annotatedData):
        if i < barrier_low:
            i += 1
            continue
        hits, missed, falseAlarm, totalBoxes, falseAlarms, distances = evaluateImage(annotatedData[i].boundingBoxes, predictedData[i].boundingBoxes)
        totalHits[0] += hits[0]
        totalHits[1] += hits[1]
        totalHits[2] += hits[2]
        totalDistances[0] += distances[0]
        totalDistances[1] += distances[1]
        totalMissed += missed
        totalFalseAlarms += falseAlarm
        totalObjects += totalBoxes
        if partialOutputs:
            print('File{:s} Hits:0.5>{:d} 0.3>{:d} 0.1>{:d} Missed:{:d} FalseAlarms:{:d} Objects:{:d}'.format(annotatedData[i].filename, hits[0], hits[1], hits[2], missed, falseAlarm, totalBoxes))
        # image = cv2.imread(PATH_TO_IMAGE_ROOT_DIR + annotatedData[i].filename)
        # for predictedBox in falseAlarms:
        #     cv2.rectangle(image, (predictedBox.x, predictedBox.y), (predictedBox.x + predictedBox.width, predictedBox.y + predictedBox.height), (0, 255, 0), 3)
        # cv2.imshow("test",image)
        # cv2.waitKey()
        i += 1
        precision = 0
        if totalHits[0] + totalHits[1] + totalHits[2] + totalFalseAlarms > 0:
            precision = float(totalHits[0] + totalHits[1] + totalHits[2])/(totalHits[0] + totalHits[1] + totalHits[2] + totalFalseAlarms) * 100
        recall = 0
        if totalHits[0] + totalHits[1] + totalHits[2] + totalMissed > 0:
            recall =  float(totalHits[0] + totalHits[1] + totalHits[2])/(totalHits[0] + totalHits[1] + totalHits[2] + totalMissed) * 100
        if i >= barrier_high:
            break
    print('\n' + modelName + ' - Total: Hits: 0.5>{:d} 0.3>{:d} 0.1>{:d} Distances: 0.3<{:d} 0.5<{:d} Missed:{:d} FalseAlarms:{:d} Objects:{:d} Precision:{:.1f}% Recall:{:.1f}%'.format(
        totalHits[0], totalHits[1], totalHits[2], totalDistances[0], totalDistances[1], totalMissed, totalFalseAlarms, totalObjects, precision, recall))


def loadAndProcess(firstVideo, modelName):
    if firstVideo:
        ANNOTATIONS_FILE_NAME = 'tracks_1_300.xml'
        barrier_low = 200
        barrier_high = 250
    else:
        ANNOTATIONS_FILE_NAME = 'deformabilityAnnotations.xml'
        barrier_low = 50
        barrier_high = 100
        modelName += 'SecondVideo'
    fileNamePredicted = 'C:\\GitHubCode\\phd\\ImageCytometry\\src\\XML\\' + ANNOTATIONS_FILE_NAME
    fileNameAnnotated = 'D:\\BigData\\cellinfluid\\Annotations\\PredictedAnnotations\\tracks_1_300_' + modelName + '.xml'
    annotatedData = []
    predictedData = []
    XMLRead.readXML(fileNamePredicted, annotatedData)
    XMLRead.readXML(fileNameAnnotated, predictedData)
    processImages(annotatedData, predictedData, barrier_low, barrier_high, modelName, False)

loadAndProcess(True, 'model21032019-200NoBackground')
loadAndProcess(True, 'model21032019_02-200NoBackground')
loadAndProcess(True, 'model08042019-200NoBackground')

loadAndProcess(False, 'model21032019-200NoBackground')
loadAndProcess(False, 'model21032019_02-200NoBackground')
loadAndProcess(False, 'model08042019-200NoBackground')

loadAndProcess(True, 'model21032019-250')
loadAndProcess(True, 'model21032019_02-250')
loadAndProcess(True, 'model08042019-250')

loadAndProcess(False, 'model21032019-250')
loadAndProcess(False, 'model21032019_02-250')
loadAndProcess(False, 'model08042019-250')

loadAndProcess(True, 'model21032019-250And50')
loadAndProcess(True, 'model21032019_02-250And50')
loadAndProcess(True, 'model08042019-250And50')

loadAndProcess(False, 'model21032019-250And50')
loadAndProcess(False, 'model21032019_02-250And50')
loadAndProcess(False, 'model08042019-250And50')





