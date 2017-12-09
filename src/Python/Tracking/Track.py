from BoundingBox import *
import math
import cv2
import numpy as np

class Track:
    def __init__(self, bounding_boxes, track=None):
        if track is not None:
            self.bounding_boxes = []
            for bb in track.bounding_boxes:
                self.bounding_boxes.append(BoundingBox(bb.x, bb.y, 1 , bb.frame_index,0,0))
        elif bounding_boxes is not None:
            self.bounding_boxes = []
            for bb in bounding_boxes:
                self.bounding_boxes.append(BoundingBox(bb[0], bb[1], bb[2], bb[3], 0, 0))
        self.speed = 0
        self.angle = 0
        self.vector = (0, 0)
        self.kalman = cv2.KalmanFilter(4, 2)
        self.kalman.measurementMatrix = np.array([[1, 0, 0, 0], [0, 1, 0, 0]], np.float32)
        self.kalman.transitionMatrix = np.array([[1, 0, 1, 0], [0, 1, 0, 1], [0, 0, 1, 0], [0, 0, 0, 1]], np.float32)
        self.kalman.processNoiseCov = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]], np.float32) * 0.03

        for bb in self.bounding_boxes:
            mp = np.array([[np.float32(bb.x)], [np.float32(bb.y)]])
            self.kalman.correct(mp)
            self.kalman.predict()

    def add_point_to_end(self, bounding_box):
        self.bounding_boxes.append(bounding_box)

    # add all bb from other Track to the end of this track
    def merge_tracks(self, other):
        for bb in other.bounding_boxes:
            self.bounding_boxes.append(bb)
            mp = np.array([[np.float32(bb.x)], [np.float32(bb.y)]])
            self.kalman.correct(mp)
            self.kalman.predict()

    def compute_speed(self):
        speed = 0
        count = 0
        for index in range(len(self.bounding_boxes) - 1):
            speed += get_distance(self.bounding_boxes[index], self.bounding_boxes[index + 1])
            count += 1
        self.speed = speed / count

    def compute_avg_vector(self):
        sum_x = 0
        sum_y = 0
        count = 0
        for index in range(len(self.bounding_boxes) - 1):
            sum_x += self.bounding_boxes[index].x - self.bounding_boxes[index + 1].x
            sum_y += self.bounding_boxes[index].x - self.bounding_boxes[index + 1].x
            count += 1
        self.vector = (sum_x / count, sum_y / count)

    def kalman_predict(self, count):
        for i in range(count):
            frame = self.bounding_boxes[-1].frame_index + 1
            predict = self.kalman.predict()
            self.bounding_boxes.append(BoundingBox(int(predict[0]), int(predict[1]),1,frame,0,0))
            mp = np.array([[np.float32(predict[0])], [np.float32(predict[1])]])
            self.kalman.correct(mp)



    def __str__(self):
        output = ""
        for bb in self.bounding_boxes:
            output = output + str(bb)
        return output


    def __eq__(self, other):
        if len(self.bounding_boxes) != len(other.bounding_boxes):
            return False

        for index in range(len(self.bounding_boxes)):
            bb = self.bounding_boxes[index]
            other_bb = other.bounding_boxes[index]
            if bb != other_bb:
                return False
        return True


class MergeTracks:

    def __init__(self, first_track, second_track):
        self.first_track = first_track
        self.second_track = second_track
        self.sum = -1

    def mean_squared_error(self):
        sum_pom = 0
        first_track_last = self.first_track.bounding_boxes[-1]
        first_track_second_last = self.first_track.bounding_boxes[-2]
        diff = self.second_track.bounding_boxes[0].frame_index - self.first_track.bounding_boxes[-1].frame_index
        for index in range(len(self.second_track.bounding_boxes)):
            # odhadnut bod na indexe
            x = first_track_last.x + (index + diff) * (first_track_last.x - first_track_second_last.x)
            y = first_track_last.y + (index + diff) * (first_track_last.y - first_track_second_last.y)
            # skutocny bod na indexe
            x2 = self.second_track.bounding_boxes[index].x
            y2 = self.second_track.bounding_boxes[index].y
            sum_pom += (abs(x - x2) + abs(y - y2))**2
        self.sum = sum_pom / len(self.second_track.bounding_boxes)


def get_distance(bb1, bb2):
    return math.sqrt((bb1.x - bb2.x) ** 2 + (bb1.y - bb2.y) ** 2)