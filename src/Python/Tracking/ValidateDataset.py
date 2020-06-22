import tensorflow as tf
import Dataset
import numpy as np
import XMLRead
import cv2

session = tf.Session()

dataset = Dataset.Dataset()
size = 31
channels = 5
rots = 4
trackingDataset = dataset.loadFromDataset('C:\\GitHubCode\\phd\\ImageCytometry\\src\\TFRecord\\tracking\\trainTracking250SimulationMatrix31AnnotatedFixedRots.record', size, channels)


file = open("outputRots.txt", "w")

for sampleId in range(rots):
    for l in range(channels):
        for i in range(len(trackingDataset.features[sampleId])):
            for k in range(len(trackingDataset.features[sampleId][i])):
                file.write(str(trackingDataset.features[sampleId][i][k][l]) + ' ')
            file.write('\n')
        file.write('\n')
    file.write('Feature values:' + str(trackingDataset.features[sampleId][int(size / 2)][int(size / 2)][0]) + ' ' + str(
        trackingDataset.features[sampleId][int(size / 2)][int(size / 2)][1]) + '\n')
    file.write('Response value:' + str(trackingDataset.response[sampleId][0]) + ' '
               + str(trackingDataset.response[sampleId][1]) + '\n')
    file.write('X: ' + str(trackingDataset.x[sampleId]) + ' Y: ' + str(trackingDataset.y[sampleId]) + ' Width: ' +
               str(trackingDataset.width[sampleId]) + ' Height: ' + str(trackingDataset.height[sampleId]) + '\n')


file.close()

# def validation():
#     for id in range(int(len(trackingDataset.features) / 4)):
#         copy = trackingDataset.features[id * rots]
#         for rotId in range(1, 3):
#             copy = np.rot90(copy)
#             for i in range(len(trackingDataset.features[0])):
#                 for k in range(len(trackingDataset.features[0][0])):
#                     if abs(trackingDataset.features[id * rots + rotId][i][k][rotId % 2]) != abs(copy[i][k][0]):
#                         print(str(id) + " " + str(rotId) + " " + str(i) + " " + str(k) + " " + str(0) + " " +
#                               str(trackingDataset.features[id * rots + rotId][i][k][rotId % 2]) + " " + str(copy[i][k][0]))
#                     if abs(trackingDataset.features[id * rots + rotId][i][k][(rotId + 1) % 2]) != abs(copy[i][k][1]):
#                         print(str(id) + " " + str(rotId) + " " + str(i) + " " + str(k) + " " + str(1) + " " +
#                               str(trackingDataset.features[id * rots + rotId][i][k][(rotId + 1) % 2]) + " " + str(copy[i][k][1]))
#
#                     if trackingDataset.features[id * rots + rotId][i][k][2] != copy[i][k][2]:
#                         print(str(id) + " " + str(rotId) + " " + str(i) + " " + str(k) + " " + str(2) + " " +
#                               str(trackingDataset.features[id * rots + rotId][i][k][rotId % 2]) + " " + str(copy[i][k][2]))
#
#                     if abs(trackingDataset.features[id * rots + rotId][i][k][rotId % 2 + 3]) != abs(copy[i][k][3]):
#                         print(str(id) + " " + str(rotId) + " " + str(i) + " " + str(k) + " " + str(3) + " " +
#                               str(trackingDataset.features[id * rots + rotId][i][k][rotId % 2 + 3]) + " " + str(copy[i][k][3]))
#                     if abs(trackingDataset.features[id * rots + rotId][i][k][(rotId + 1) % 2 + 3]) != abs(copy[i][k][4]):
#                         print(str(id) + " " + str(rotId) + " " + str(i) + " " + str(k) + " " + str(4) + " " +
#                               str(trackingDataset.features[id * rots + rotId][i][k][rotId % 2 + 3]) + " " + str(copy[i][k][4]))
#
#
# validation()

# PATH_TO_BACKGROUND = 'D:\\BigData\\cellinfluid\\bunkyObrazkyTiff\\background.png'
# image = cv2.imread(PATH_TO_BACKGROUND)
# annotatedData = []
# XMLRead.readXML('C:\\GitHubCode\\phd\\ImageCytometry\\src\\XML\\tracks_1_300.xml', annotatedData)
# tracks = XMLRead.initTracks(annotatedData)
# velocities = np.zeros((720, 1280, 2), np.float32)
# for currentKey, currentTrack in tracks.items():
#     for curBoxIndex in range(len(currentTrack.boundingBoxes)):
#         boundingBox = currentTrack.boundingBoxes[curBoxIndex]
#         if curBoxIndex > 0:
#             curPrevBoundingBox = currentTrack.boundingBoxes[curBoxIndex - 1]
#             fillingVelocityX = (boundingBox.x - curPrevBoundingBox.x) / float(
#                 abs(curPrevBoundingBox.frameId - boundingBox.frameId))
#             fillingVelocityY = (boundingBox.y - curPrevBoundingBox.y) / float(
#                 abs(curPrevBoundingBox.frameId - boundingBox.frameId))
#             for indexY in range(boundingBox.height):
#                 for indexX in range(boundingBox.width):
#                     if velocities[boundingBox.y + indexY][boundingBox.x + indexX][0] < fillingVelocityX:
#                         velocities[boundingBox.y + indexY][boundingBox.x + indexX][0] = fillingVelocityX
#                     if velocities[boundingBox.y + indexY][boundingBox.x + indexX][1] < fillingVelocityY:
#                         velocities[boundingBox.y + indexY][boundingBox.x + indexX][1] = fillingVelocityY
# for i in range(720):
#     for k in range(1280):
#         if velocities[i][k][0] > 0 or velocities[i][k][1] > 0:
#             image[i][k] = [0, 0, 255]
#
# cv2.imwrite('FlowMatrix.png', image)
