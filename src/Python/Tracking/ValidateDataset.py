import tensorflow as tf
import Dataset
import numpy as np
import XMLRead
import cv2

session = tf.Session()

dataset = Dataset.Dataset()
trackingDataset = dataset.loadFromDataset('C:\\GitHubCode\\phd\\ImageCytometry\\src\\TFRecord\\tracking\\trainTracking250SimulationMatrixFixed.record')

for l in range(5):
    for i in range(len(trackingDataset.features[0])):
        for k in range(len(trackingDataset.features[0][i])):
            print(trackingDataset.features[0][i][k][l], end=" ")
        print()
    print()

print('X: ' + str(trackingDataset.x[0]) + ' Y: ' + str(trackingDataset.y[0]) + ' Width: ' + str(trackingDataset.width[0]) + ' Height: ' + str(trackingDataset.height[0]))
print([trackingDataset.features[0][30][30][0], trackingDataset.features[0][30][30][1]])
print(trackingDataset.response[0])

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
