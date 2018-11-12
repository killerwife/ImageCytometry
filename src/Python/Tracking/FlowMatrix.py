
import numpy as np
import math
import cv2
import random

def print_flow_matrix(flow_matrix):
    new_list = flow_matrix.copy()
    #list2 = rotate(new_list)
    #print_matrix(list2)
    print_matrix(new_list)


def calculate_flow_matrix(flow_matrix):
    """
    Funkcia dopocita tokovu maticu.
    :param flow_matrix: tokova matica
    :return: dopocitana tokova matica
    """
    old_matrix = flow_matrix.copy()
    for x in range(len(flow_matrix)):
        for y in range(len(flow_matrix[x])):
            if flow_matrix[x][y][0] == -1:
                # find first point on top and bot from current
                # print(str(x) + ' , ' + str(y))
                loop = True
                left_move = False
                left = 0
                right = 0
                distance = 0
                right_move = False
                x_index = x
                y_index = y
                while loop:
                    if left_move:
                        left += 1
                        x_index = x - left
                    elif right_move:
                        right += 1
                        x_index = x + right
                    y_index = y
                    top = None
                    top_x = -1
                    top_y = -1
                    bot = None
                    bot_x = -1
                    bot_y = -1
                    #print('loop=' + str(x_index) + ' , ' + str(y_index))
                    while y_index < len(flow_matrix[x]) - 1:
                        y_index += 1
                        if flow_matrix[x_index][y_index][0] >= 0:
                            #print('x=' + str(x_index) + ' y=' + str(y_index))
                            #print(flow_matrix[x_index][y_index])
                            bot = flow_matrix[x_index][y_index][1]
                            bot_x = x_index
                            bot_y = y_index
                            break
                    y_index = y
                    while y_index > 0:
                        y_index -= 1
                        if flow_matrix[x_index][y_index][0] >= 0:
                            top = flow_matrix[x_index][y_index][1]
                            top_x = x_index
                            top_y = y_index
                            break
                    if top is not None and bot is not None:
                        distance_top = get_distance([x, y], [top_x, top_y])
                        distance_bot = get_distance([x, y], [bot_x, bot_y])
                        x_top = 1/distance_top * top[0]
                        y_top = 1/distance_top * top[1]
                        x_bot = 1/distance_bot * bot[0]
                        y_bot = 1/distance_bot * bot[1]
                        sum_distance = (1 / distance_top + 1/ distance_bot)

                        flow_matrix[x][y][0] = 20
                        flow_matrix[x][y][1] = [(x_top + x_bot) / sum_distance, (y_top + y_bot) / sum_distance]
                        if (x_top + x_bot) / sum_distance > 100 or (y_top + y_bot) / sum_distance > 100:
                            print('a')
                            print([(x_top + x_bot) / sum_distance, (y_top + y_bot) / sum_distance])
                        loop = False
                    elif top is not None:
                        flow_matrix[x][y][0] = 20
                        flow_matrix[x][y][1] = [top[0], top[1]]
                        if top[0] > 100 or top[1] > 100:
                            print('b')
                            print([top[0], top[1]])
                        loop = False
                    elif bot is not None:
                        flow_matrix[x][y][0] = 20
                        flow_matrix[x][y][1] = [bot[0], bot[1]]
                        if bot[0] > 100 or bot[1] > 100:
                            print([bot[0], bot[1]])
                        loop = False
                    else:
                        if x - left == 0:
                            right_move = True
                        elif x + right == len(flow_matrix) - 1:
                            left_move = True
                        elif right_move:
                            right_move = False
                            left_move = True
                        elif left_move:
                            right_move = False
                            left_move = True
                        else:
                            right_move = True




def print_points(flow_matrix, cols=1280, rows=720):
    calculate_flow_matrix(flow_matrix)
    #print_matrix(flow_matrix)
    img = np.zeros((rows, cols, 3), np.uint8)
    for row_index in range(len(flow_matrix)):
        for col_index in range(len(flow_matrix[row_index])):
            if flow_matrix[row_index][col_index ][0] >= 0:
                x = flow_matrix[row_index][col_index ][1][0]
                y = flow_matrix[row_index][col_index][1][1]
                cv2.line(img, (row_index, col_index), (row_index, col_index), (0, 0, 255))
            if flow_matrix[row_index][col_index ][0] == 20:
                cv2.line(img, (row_index, col_index), (row_index, col_index), (0, 255, 0))
    cv2.imshow("Pred", img)
    k = cv2.waitKey(0)
    if k == ord('q'):
        cv2.destroyAllWindows()


def print_flow_matrix_default(flow_matrix, cols=1280, rows=720):
    img = cv2.imread('input.tiff')
    #img = np.zeros((rows, cols, 3), np.uint8)
    for row_index in range(len(flow_matrix)):
            for col_index in range(len(flow_matrix[row_index])):
                if flow_matrix[row_index][col_index][0] >= 0:
                    color = (random.randint(120, 250), random.randint(120, 250), random.randint(120, 250))
                    red = (0, 0, 255)
                    x = row_index + flow_matrix[row_index][col_index][1][0]
                    y = col_index + flow_matrix[row_index][col_index][1][1]
                    cv2.line(img, (int(row_index), int(col_index)), (int(x), int(y)), red)

    cv2.imwrite('input2.png', img)
    cv2.imshow("Pred", img)
    k = cv2.waitKey(0)
    if k == ord('q'):
        cv2.destroyAllWindows()


def print_matrix(flow_matrix, cols=1280, rows=720):
    img = np.zeros((rows, cols, 3), np.uint8)
    for row_index in range(len(flow_matrix)):
        if row_index != 0 and row_index % 7 == 0:
            for col_index in range(len(flow_matrix[row_index])):
                if col_index != 0 and col_index % 7 == 0:
                    start_y = col_index
                    start_x = row_index
                    x = flow_matrix[row_index][col_index][1][0]
                    y = flow_matrix[row_index][col_index][1][1]
                    radians = math.atan2(y, x)
                    end_y = start_y + y
                    end_x = start_x + x
                    cv2.line(img, (int(start_x), int(start_y)), (int(end_x), int(end_y)), (0, 0, 255))

    cv2.imshow("Flow Matrix", img)
    k = cv2.waitKey(0)
    if k == ord('q'):
        cv2.destroyAllWindows()


def save_to_file(flow_matrix):
    """
    Ulozi tokovu maticu do suboru ako csv subor.
    :param flow_matrix:
    :return:
    """
    string = ''
    for row_index in range(len(flow_matrix)):
        row = ''
        for col_index in range(len(flow_matrix[row_index])):
            if flow_matrix[row_index][col_index][0] > 0:
                row += str(flow_matrix[row_index][col_index][1][0]) + ','
                row += str(flow_matrix[row_index][col_index][1][1])
            else:
                row += '0,0'
            row += ';'
        string += row + '\n'

    with open('flow_matrix.csv', 'w') as file:
        file.write(string)


def load_from_file(file_name):
    """
    Funkcia nacita tokovu maticu zo suboru.
    :param file_name:
    :return:
    """
    flow_matrix = []
    with open(file_name) as file:
        lines = file.read().splitlines()
        for line in lines:
            print(line)
            row = []
            items = line.split(';')
            for item in items:
                data = item.split(',')
                print(data)
                if data == '':
                    break
                x = float(data[0])
                y = float(data[1])
                if x == 0 and y == 0:
                    row.append([-1, [x, y]])
                else:
                    row.append([1, [x, y]])
            flow_matrix.append(row)
    return flow_matrix

def rotate(old_matrix):
    w = len(old_matrix)
    h = len(old_matrix[0])
    new_matrix = [[0 for x in range(w)] for y in range(h)]
    for row_index in range(len(old_matrix)):
        for col_index in range(len(old_matrix[row_index])):
            new_matrix[col_index][row_index] = old_matrix[row_index][col_index]
    return new_matrix


def get_distance(a, b):
    return math.sqrt((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2)
