import cv2
import Definitions
import math


class CellPosition(object):
    def __init__(self):
        self.id = 0
        self.x = 0
        self.y = 0
        self.z = 0

    def print(self):
        print('ID: ' + str(self.id) + ' X: ' + str(self.x) + ' Y: ' + str(self.y) + ' Unk: ' + str(self.z))


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
    def __init__(self, x=0, y=0, velocityX=0, velocityY=0):
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
                # stupid flow matrix in file can be bigger
                if entry.x >= (self.width / 3) or entry.y >= (self.height / 3):
                    continue
                entry.z = int(lineSplit[2])
                # conversion of units from 3ms
                entry.velocityX = float(lineSplit[3]) * 1000
                entry.velocityY = float(lineSplit[4]) * 1000
                entry.velocityZ = float(lineSplit[5]) * 1000
                newData.append(entry)

        # get all data for each point - we only have data for every third
        temp3DArray = [[[] for x in range(self.width)] for y in range(self.height)]
        for data in newData:
            temp3DArray[data.y * 3][data.x * 3].append(data)
            temp3DArray[data.y * 3 + 1][data.x * 3].append(data)
            temp3DArray[data.y * 3 + 2][data.x * 3].append(data)
            temp3DArray[data.y * 3][data.x * 3 + 1].append(data)
            temp3DArray[data.y * 3 + 1][data.x * 3 + 1].append(data)
            temp3DArray[data.y * 3 + 2][data.x * 3 + 1].append(data)
            temp3DArray[data.y * 3][data.x * 3 + 2].append(data)
            temp3DArray[data.y * 3 + 1][data.x * 3 + 2].append(data)
            temp3DArray[data.y * 3 + 2][data.x * 3 + 2].append(data)

        # calculate 2D flow matrix - iterate over indices cos of incomplete data in columns - just fill 0
        for rowIter in range(len(temp3DArray)):
            for colIter in range(len(temp3DArray[rowIter])):
                maxVelocityX = 0
                maxVelocityY = 0
                for pointData in temp3DArray[rowIter][colIter]:
                    if maxVelocityX < pointData.velocityX:
                        maxVelocityX = pointData.velocityX
                    if maxVelocityY < pointData.velocityY:
                        maxVelocityY = pointData.velocityY
                self.data[rowIter][colIter] =\
                    FlowMatrixData2D(colIter, rowIter, maxVelocityX, maxVelocityY)

    def convertToOldArrayType(self):
        output = [[[1, []] for j in range(self.height)] for i in range(self.width)]
        for row in self.data:
            for column in row:
                output[column.x][column.y][1] = [column.velocityX, column.velocityY]
        return output

    def oldFlowMatrix(self, tracks, unresolved_from_tracking):
        def get_distance(a, b):
            return math.sqrt((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2)

        def create_flow_matrix(x, y):
            # vytvorenie prazdnej matice o velkosti XxY
            # print('Creating matrix...')
            matrix = [[[-1, []] for j in range(y)] for i in range(x)]
            return matrix

        def calculate_flow_matrix(flow_matrix, tracks):
            print('calculating flow matrix')
            points = 0
            # vypocet vektorov pre body, ktore su v trackoch
            for i in range(len(tracks)):
                if not tracks[i]:
                    continue
                for j in range(len(tracks[i]) - 1):
                    if not tracks[j]:
                        continue
                    points += 1
                    x1 = tracks[i][j][0]
                    x2 = tracks[i][j + 1][0]
                    x = x2 - x1
                    y1 = tracks[i][j][1]
                    y2 = tracks[i][j + 1][1]
                    y = y2 - y1
                    calculate_point(flow_matrix, x, y, tracks[i][j][0], tracks[i][j][1])

                # nastavenie vektora posledneho bodu v tracku
                points += 1
                calculate_point(flow_matrix, x, y, tracks[i][-1][0], tracks[i][-1][1])

        def calculate_point(flow_matrix, vector_x, vector_y, cor_x, cor_y):
            count = flow_matrix[cor_x][cor_y][0]

            if count == -1:
                flow_matrix[cor_x][cor_y][1] = [vector_x, vector_y]
                flow_matrix[cor_x][cor_y][0] = 1
            else:
                # vypocet priemerneho vektora
                avg_x = ((count * flow_matrix[cor_x][cor_y][1][0]) + vector_x) / (count + 1)
                avg_y = ((count * flow_matrix[cor_x][cor_y][1][1]) + vector_y) / (count + 1)
                flow_matrix[cor_x][cor_y][1] = [avg_x, avg_y]
                flow_matrix[cor_x][cor_y][0] += 1

        def resolve_flow_matrix(flow_matrix, unresolved_from_tracking):
            # doplnit o body, ktore nie su v trackoch
            print('resolving flow matrix: calculating vector for not resolved points')
            no = 0
            for k in range(len(unresolved_from_tracking)):
                x = unresolved_from_tracking[k][0]
                y = unresolved_from_tracking[k][1]
                if flow_matrix[x][y][0] == -1:
                    resolve_point(flow_matrix, x, y, 1)
            #     else:
            #         no += 1
            # print('Pocet bodov, ktore sa nedopocitavaju: '+ str(no))
            print('\tdone')

        def resolve_point(flow_matrix, cor_x, cor_y, index):
            max_x = len(flow_matrix) - 1
            max_y = len(flow_matrix[0]) - 1

            range_end = index * 2 + 1
            range_start = -1 * index
            candidates_vectors = []
            candidates_cor = []
            sum_x = 0
            sum_y = 0
            sum_distance = 0

            # rozsah pre y
            for j in range(range_start, (range_end + range_start)):
                # pocitame cely riadok

                if j == range_start or j == range_start * (-1):
                    for i in range(range_start, (range_end + range_start)):
                        if (max_x >= cor_x + i >= 0) and (max_y >= cor_y + j >= 0):  # overit suradnice
                            # if ((cor_x + i <= max_x ) and (cor_y + j <= max_y  )):
                            # print(str(cor_x + i )+ ' ' +str(cor_y + j))
                            if flow_matrix[cor_x + i][cor_y + j][0] > 0:
                                candidates_vectors.append(flow_matrix[cor_x + i][cor_y + j][1])
                                candidates_cor.append([cor_x + i, cor_y + j])
                else:
                    if max_x >= cor_x + range_start >= 0 and max_y >= cor_y + j >= 0:
                        if flow_matrix[cor_x + range_start][cor_y + j][0] > 0:
                            candidates_vectors.append(flow_matrix[cor_x + range_start][cor_y + j][1])
                            candidates_cor.append([cor_x + range_start, cor_y + j])
                        elif cor_x - range_start <= max_x and cor_y + j <= max_y:
                            if flow_matrix[cor_x - range_start][cor_y + j][0] > 0:
                                candidates_vectors.append(flow_matrix[cor_x - range_start][cor_y + j][1])
                                candidates_cor.append([cor_x - range_start, cor_y + j])

            if len(candidates_vectors) == 0:
                resolve_point(flow_matrix, cor_x, cor_y, index + 1)
            else:
                for s in range(len(candidates_vectors)):
                    # nascitanie suradnic
                    distance = get_distance([cor_x, cor_y], candidates_cor[s])
                    sum_x += (1 / distance) * candidates_vectors[s][0]
                    sum_y += (1 / distance) * candidates_vectors[s][1]  # # nascitanie menovatela
                    sum_distance += (1 / distance)
                    flow_matrix[cor_x][cor_y][1] = [sum_x / sum_distance, sum_y / sum_distance]
                    flow_matrix[cor_x][cor_y][0] = 0

        flow_matrix = create_flow_matrix(int(self.width), int(self.height))
        calculate_flow_matrix(flow_matrix, tracks)
        resolve_flow_matrix(flow_matrix, unresolved_from_tracking)
        return flow_matrix


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
    cells = readCellData(Definitions.DATA_ROOT_DIRECTORY + Definitions.FILE_ID_NAME)
    readCellPositions(Definitions.DATA_ROOT_DIRECTORY + Definitions.POSITIONS_FOLDER + '\\', cells, Definitions.POSITION_FILE_PROTO)
    for cell in cells:
        cell.print()

    validateCellPositions(cells)

    matrix = FlowMatrix(1280, 720)
    matrix.readFlowMatrix(Definitions.DATA_ROOT_DIRECTORY + Definitions.FLOW_MATRIX_FILE)


if __name__ == "__main__":
    main()



