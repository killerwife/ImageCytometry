import cv2


class CellPosition(object):
    def __init__(self):
        self.id = 0
        self.x = 0
        self.y = 0
        self.z = 0

    def print(self):
        print('ID: ' + str(self.id) + ' X: ' + str(self.x) + ' Y: ' + str(self.y)+ ' Unk: ' + str(self.z))

class CellData(object):
    def __init__(self):
        self.id = 0
        self.radius = float(0)
        self.variant = 0
        self.cellPositions = []

    def print(self):
        print('ID: ' + str(self.id) + ' Radius: ' + str(self.radius) + ' Variant: ' + str(self.variant))
        for cellPos in self.cellPositions:
            cellPos.print()


class FlowMatrixData(object):
    def __init__(self):
        self.x = 0
        self.y = 0
        self.z = 0
        self.velocityX = 0
        self.velocityY = 0
        self.velocityZ = 0


DATA_ROOT_DIRECTORY = 'D:\\BigData\\cellinfluid\\TrackingData\\'
FILE_ID_NAME = 'cells_IDs.dat'
POSITIONS_FOLDER = 'cell_positions'
POSITION_FILE_PROTO = 'cell_position_{:d}.dat'
FLOW_MATRIX_FILE = 'V1_flow_matrix_simNo_1.txt'


def readCellData(filename):
    cells = []
    with open(filename) as file:
        for line in file:
            lineSplit = line.split()
            cellData = CellData()
            cellData.id = int(lineSplit[0])
            cellData.radius = float(lineSplit[1])
            cellData.variant = int(lineSplit[2])
            cells.append(cellData)
    return cells


def readCellPositions(directory, cells):
    for cell in cells:
        print(directory + POSITION_FILE_PROTO.format(cell.id))
        with open(directory + POSITION_FILE_PROTO.format(cell.id)) as file:
            for line in file:
                lineSplit = line.split()
                if len(lineSplit) < 3:
                    continue
                cellPos = CellPosition()
                cellPos.id = int(lineSplit[0])
                cellPos.x = int(float(lineSplit[1]) * 3)
                cellPos.y = int(float(lineSplit[2]) * 3)
                cellPos.z = int(float(lineSplit[3]) * 3)
                cell.cellPositions.append(cellPos)


def validateCellPositions(cells):
    image = cv2.imread('D:\\BigData\\cellinfluid\\bunkyObrazkyTiff\\1-50\\video2359_0001.tiff')
    for cell in cells:
        for position in cell.cellPositions:
            image[image.shape[0] - position.y, position.x] = [0, 0, 255]

    cv2.imshow('Test', image)
    cv2.waitKey()


def readFlowMatrix(filepath):
    flowMatrix = []
    with open(filepath) as file:
        for line in file:
            lineSplit = line.split()
            if len(lineSplit) < 6:
                continue
            data = FlowMatrixData()
            data.x = int(lineSplit[0])
            data.y = int(lineSplit[1])
            data.z = int(lineSplit[2])
            data.velocityX = float(lineSplit[3])
            data.velocityY = float(lineSplit[4])
            data.velocityZ = float(lineSplit[5])
            flowMatrix.append(data)

    return flowMatrix



cells = readCellData(DATA_ROOT_DIRECTORY + FILE_ID_NAME)
readCellPositions(DATA_ROOT_DIRECTORY + POSITIONS_FOLDER + '\\', cells)
for cell in cells:
    cell.print()

validateCellPositions(cells)

readFlowMatrix(DATA_ROOT_DIRECTORY + FLOW_MATRIX_FILE)



