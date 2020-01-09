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


class FlowMatrixData2D(object):
    def __init__(self):
        self.x = 0
        self.y = 0
        self.velocityX = 0
        self.velocityY = 0

    def __init__(self, x, y, velocityX = 0, velocityY = 0):
        self.x = x
        self.y = y
        self.velocityX = velocityX
        self.velocityY = velocityY


class FlowMatrixData3D(object):
    def __init__(self):
        self.x = 0
        self.y = 0
        self.z = 0
        self.velocityX = 0
        self.velocityY = 0
        self.velocityZ = 0


class Vector2D(object):
    def __init__(self):
        self.x = 0
        self.y = 0


class FlowMatrix(object):
    def __init__(self, width, height):
        self.width = width
        self.height = height
        self.data = [[FlowMatrixData2D() for x in range(width)] for y in range(height)]

    def readFlowMatrix(self, filepath):
        newData = []  # load 3D data
        with open(filepath) as file:
            for line in file:
                lineSplit = line.split()
                if len(lineSplit) < 6:
                    continue
                entry = FlowMatrixData3D()
                entry.x = int(lineSplit[0])
                entry.y = int(lineSplit[1])
                entry.z = int(lineSplit[2])
                entry.velocityX = float(lineSplit[3])
                entry.velocityY = float(lineSplit[4])
                entry.velocityZ = float(lineSplit[5])
                newData.append(entry)

        # get all data for each point
        temp3DArray = [[[] for x in range(self.width)] for y in range(self.height)]
        for data in newData:
            temp3DArray[entry.y][entry.x].append(data)

        # calculate 2D flow matrix
        for row in temp3DArray:
            for column in row:
                maxVelocityX = 0
                maxVelocityY = 0
                for pointData in column:
                    if maxVelocityX < pointData.velocityX:
                        maxVelocityX = pointData.velocityX
                    if maxVelocityY < pointData.velocityY:
                        maxVelocityY = pointData.velocityY
                self.data[column[0].y][column[0].x] =\
                    FlowMatrixData2D(column[0].x, column[0].y, maxVelocityX, maxVelocityY)

    def convertToOldArrayType(self):
        output = [[[1, []] for j in range(self.width)] for i in range(self.height)]
        for row in self.data:
            for column in self.data:
                output[column.y][column.x][1] = [column.velocityX, column.velocityY]


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


def readCellPositions(directory, cells, positionFileProto):
    for cell in cells:
        print(directory + positionFileProto.format(cell.id))
        with open(directory + positionFileProto.format(cell.id)) as file:
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


def main():
    DATA_ROOT_DIRECTORY = 'D:\\BigData\\cellinfluid\\TrackingData\\'
    FILE_ID_NAME = 'cells_IDs.dat'
    POSITIONS_FOLDER = 'cell_positions'
    POSITION_FILE_PROTO = 'cell_position_{:d}.dat'
    FLOW_MATRIX_FILE = 'V1_flow_matrix_simNo_1.txt'
    cells = readCellData(DATA_ROOT_DIRECTORY + FILE_ID_NAME)
    readCellPositions(DATA_ROOT_DIRECTORY + POSITIONS_FOLDER + '\\', cells, POSITION_FILE_PROTO)
    for cell in cells:
        cell.print()

    validateCellPositions(cells)

    matrix = FlowMatrix(1280, 720)
    matrix.readFlowMatrix(DATA_ROOT_DIRECTORY + FLOW_MATRIX_FILE)


if __name__== "__main__":
  main()



