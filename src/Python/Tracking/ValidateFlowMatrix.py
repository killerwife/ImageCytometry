import Definitions
import CellDataReader
import cv2
# import XMLRead
# import SimpleTracking
# import random

PATH_TO_BACKGROUND = 'D:\\BigData\\cellinfluid\\bunkyObrazkyTiff\\background.png'
image = cv2.imread(PATH_TO_BACKGROUND)

flowMatrixNew = CellDataReader.FlowMatrix(1280, 720, 3)
flowMatrixNew.readFlowMatrix(Definitions.DATA_ROOT_DIRECTORY + Definitions.FLOW_MATRIX_FILE)
flow_matrix = flowMatrixNew.convertToOldArrayType()

# file_name = 'tracks_1_300.xml'
# xml_dir = 'C:\\GitHubCode\\phd\\ImageCytometry\\src\\XML\\'
# flowMatrixFileName = ''
# frameCount = 300
# oldFormat = False
# flowMatrixSimulation = True
# evalAnnotatedTracks = False
# x = 1280
# y = 720
# frame_rate = 1/30
# pixel_size = 1/3
#
# unresolved_from_tracking = []
# mat = []
# src_names = []
#
# if not oldFormat:
#     annotatedData = []
#     XMLRead.readXML(xml_dir + file_name, annotatedData)
#     tracks, mat, src_names = XMLRead.parseXMLDataForTracks(annotatedData, evalAnnotatedTracks)
#
# # zmaze niektore bunky z anastroja - z matice mat !!!!!!!!!!!!!!
# random.seed(20)
# # mat = remove_one_cell_from_all_frame(mat)
# # mat = remove_one_cell_from_all_frame(mat)
# # mat = remove_some_cell_random(mat, 50)
# # mat only 200 frames
# mat = mat[:frameCount]
# parameters = '12 8 8'
# parameters = parameters.split(' ')
# dist = int(parameters[0])
# a = int(parameters[1])
# b = int(parameters[2])
# tracks, unresolved_from_tracking = SimpleTracking.predicting_tracking(dist, a, b, mat)
# annotatedData = []
# XMLRead.readXML('C:\\GitHubCode\\phd\\ImageCytometry\\src\\XML\\tracks_1_300.xml', annotatedData)
# tracks, mat, src_names = XMLRead.parseXMLDataForTracks(annotatedData, True)
# flowMatrix = CellDataReader.FlowMatrix(1280, 720, 3)
# flow_matrix = flowMatrix.oldFlowMatrix(tracks, unresolved_from_tracking)

for i in range(720):
    for k in range(1280):
        if flow_matrix[k][i][0] != -1 and (flow_matrix[k][i][1][0] != 0 or flow_matrix[k][i][1][1] != 0):
            image[i][k] = [0, 0, 255]

cv2.imwrite('FlowMatrix.png', image)

image = cv2.imread(PATH_TO_BACKGROUND)

newData = []
width = 1280
height = 720
multiplier = 3
with open(Definitions.DATA_ROOT_DIRECTORY + Definitions.FLOW_MATRIX_FILE) as file:
    for line in file:
        lineSplit = line.split()
        if len(lineSplit) < 6:
            continue
        entry = CellDataReader.FlowMatrixData3D()
        entry.x = int(lineSplit[0]) * multiplier
        entry.y = height - 1 - int(lineSplit[1]) * multiplier
        # stupid flow matrix in file can be bigger
        if entry.x + multiplier - 1 >= width or entry.y + multiplier - 1 >= height:
            continue
        entry.z = int(lineSplit[2])
        # conversion of units from 3ms
        entry.velocityX = float(lineSplit[3]) * 1000
        entry.velocityY = float(lineSplit[4]) * 1000
        entry.velocityZ = float(lineSplit[5]) * 1000
        newData.append(entry)

for data in newData:
    if data.x < 1280 and data.y < 720:
        if data.velocityX != 0 or data.velocityY != 0 or data.velocityZ != 0:
            image[data.y][data.x] = [0, 0, 255]

cv2.imwrite('FlowMatrixSecond.png', image)


