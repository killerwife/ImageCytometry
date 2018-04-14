from BoundingBox import *
import math
import cv2
import numpy as np

class Track:
    def __init__(self, bounding_boxes, track=None):
        self.speed = 0
        self.id = -1
        if track is not None:
            self.bounding_boxes = []
            self.id = track.id
            for bb in track.bounding_boxes:
                self.bounding_boxes.append(BoundingBox(bb.x, bb.y, 1 , bb.frame_index,bb.width,bb.height))
        elif bounding_boxes is not None:
            self.bounding_boxes = []
            for bb in bounding_boxes:
                self.bounding_boxes.append(BoundingBox(bb[0], bb[1], bb[2], bb[3], bb[4], bb[5]))
        self.angle = 0
        self.vector = (0, 0)
        self.speed = 0

    def add_point_to_end(self, bounding_box):
        self.bounding_boxes.append(bounding_box)

    # add all bb from other Track to the end of this track
    def merge_tracks(self, other):
        for bb in other.bounding_boxes:
            self.bounding_boxes.append(bb)

    def compute_avg_vector(self):
        sum_x = 0
        sum_y = 0
        count = 0
        for index in range(len(self.bounding_boxes) - 1):
            sum_x += self.bounding_boxes[index].x - self.bounding_boxes[index + 1].x
            sum_y += self.bounding_boxes[index].x - self.bounding_boxes[index + 1].x
            count += 1
        self.vector = (sum_x / count, sum_y / count)


    def compute_speed(self):
        if len(self.bounding_boxes) == 0:
            self.speed = 0
            return
        total_distance = 0
        count = 0
        for index in range(len(self.bounding_boxes) - 1):
            first = self.bounding_boxes[index]
            second = self.bounding_boxes[index + 1]
            distance_x = abs(second.x - first.x)
            distance_y = abs(second.y - first.y)
            total_distance += math.sqrt(distance_x**2 + distance_y**2)
            count += 1

        self.speed = total_distance / count



    def haveBB(self, array):
        for bb in self.bounding_boxes:
            if bb.x == array[0] and bb.y == array[1] and bb.frame_index == array[3]:
                return True
        return False

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
        self.distance = -1

    def mean_squared_error(self):
        sum_pom = 0
        first_track_last = self.first_track.bounding_boxes[-1]
        first_track_second_last = self.first_track.bounding_boxes[-2]
        first_track_third_last = self.first_track.bounding_boxes[-3]
        diff = self.second_track.bounding_boxes[0].frame_index - self.first_track.bounding_boxes[-1].frame_index
        for index in range(len(self.second_track.bounding_boxes)):
            # odhadnut bod na indexe
            x = first_track_last.x + (index + diff) * (first_track_last.x - first_track_second_last.x)
            y = first_track_last.y + (index + diff) * (first_track_last.y - first_track_second_last.y)
            #x = first_track_last.x + (index + diff) * ((first_track_last.x - first_track_second_last.x)/2 + (first_track_second_last.x - first_track_third_last.x)/2)
            #y = first_track_last.y + (index + diff) * ((first_track_last.y - first_track_second_last.y)/2 + (first_track_second_last.y - first_track_third_last.y)/2)
            # skutocny bod na indexe
            x2 = self.second_track.bounding_boxes[index].x
            y2 = self.second_track.bounding_boxes[index].y
            sum_pom += (abs(x - x2) + abs(y - y2))**2
        self.sum = sum_pom / len(self.second_track.bounding_boxes)

    def mean_squared_error_n_last(self,last_bb=3):
        last = last_bb
        if last_bb > len(self.first_track.bounding_boxes):
            last = len(self.first_track.bounding_boxes)

        sum_pom = 0
        actual_index = len(self.first_track.bounding_boxes) - last
        x = self.first_track.bounding_boxes[-1].x
        y = self.first_track.bounding_boxes[-1].y
        diff = self.second_track.bounding_boxes[0].frame_index - self.first_track.bounding_boxes[-1].frame_index
        for index in range(len(self.second_track.bounding_boxes) + diff):
            if actual_index >= len(self.first_track.bounding_boxes) - 2:
                actual_index = len(self.first_track.bounding_boxes) - last
            x = x + (self.first_track.bounding_boxes[actual_index + 1].x - self.first_track.bounding_boxes[actual_index].x)
            y = y + (self.first_track.bounding_boxes[actual_index + 1].y - self.first_track.bounding_boxes[actual_index].y)
            actual_index += 1
            if index >= diff:
                x2 = self.second_track.bounding_boxes[index - diff].x
                y2 = self.second_track.bounding_boxes[index - diff].y
                sum_pom += (abs(x - x2) + abs(y - y2)) ** 2


        self.sum = sum_pom / len(self.second_track.bounding_boxes)


    def mean_squared_error_alfa_n(self):
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
            sum_pom += (1/(index + 1))*((abs(x - x2) + abs(y - y2))**2)
        self.sum = sum_pom / len(self.second_track.bounding_boxes)


    def mean_squared_error_alfa_nk(self, k):
        if len(self.second_track.bounding_boxes) > k:
            return self.mean_squared_error_alfa_n()
        sum_pom = 0
        first_track_last = self.first_track.bounding_boxes[-1]
        first_track_second_last = self.first_track.bounding_boxes[-2]
        diff = self.second_track.bounding_boxes[0].frame_index - self.first_track.bounding_boxes[-1].frame_index
        for index in range(k):
            # odhadnut bod na indexe
            x = first_track_last.x + (index + diff) * (first_track_last.x - first_track_second_last.x)
            y = first_track_last.y + (index + diff) * (first_track_last.y - first_track_second_last.y)
            # skutocny bod na indexe
            x2 = self.second_track.bounding_boxes[index].x
            y2 = self.second_track.bounding_boxes[index].y
            sum_pom += (1/(index + 1))*((abs(x - x2) + abs(y - y2))**2)
        self.sum = sum_pom / len(self.second_track.bounding_boxes)



def get_distance(bb1, bb2):
    return math.sqrt((bb1.x - bb2.x) ** 2 + (bb1.y - bb2.y) ** 2)