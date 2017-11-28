from Track import *
from BoundingBox import *
import math
import cv2
import numpy as np
import random

def merge_tracks(track_array, unresolved_points, difference, radius = 12):
    print("start merging")
    tracks = create_tracks(track_array)
    before = []
    for track in tracks:
        before.append(Track(None, track))
    unresolved_points = create_bounding_boxes(unresolved_points)
    # loop all tracks
    array_for_join = []
    array_for_join_points = []
    for track in tracks:
        # for all 1 to frame difference add point and check tracks for merge
        tracks_in_radius = []
        points_in_radius = []
        for index in range(difference):
            # add new end point TODO maybe not needed
            new_x = (2 + index) * track.bounding_boxes[-1].x - (index + 1) * track.bounding_boxes[-2].x
            new_y = (2 + index) * track.bounding_boxes[-1].y - (index + 1) * track.bounding_boxes[-2].y
            frame_index = track.bounding_boxes[-1].frame_index + (index + 1)  # start from 0
            new_bb = BoundingBox(new_x, new_y, 1, frame_index, 0, 0)
            # check tracks for merge
            for track_for_merge in tracks:
                if track != track_for_merge:
                    # compare last point from first track with first point in second track
                    track2_first_p = track_for_merge.bounding_boxes[0]
                    if is_in_radius(new_bb, track2_first_p, radius) and new_bb.frame_index == track2_first_p.frame_index:
                        tracks_in_radius.append(track_for_merge)
            # check unresolved points for merge
            for point in unresolved_points:
                if is_in_radius(new_bb, point, radius) and new_bb.frame_index == point.frame_index:
                    points_in_radius.append(point)
        # add array of all tracks which can be joined to track
        array_for_join.append(tracks_in_radius)
        # add array of all points which can be joined to track
        array_for_join_points.append(points_in_radius)

    '''for index in range(len(array_for_join)):
        output = "array=" + str(tracks[index]) + " join with "
        for track in array_for_join[index]:
            output += str(track) + " and "
        #print(output)
    for index in range(len(array_for_join)):
        for track in array_for_join[index]:
            tracks[index].merge_tracks(track)
        print(tracks[index])
    for index in range(len(array_for_join_points)):
        output = str(tracks[index]) + " join with point "
        for point in array_for_join_points[index]:
            output += str(point) + " and "
        #print(output)'''
    '''newTracks, newTracks2 = join_tracks(tracks, array_for_join)
    print_track(tracks, newTracks, newTracks2)
    x = 5
    y = 10'''
    merged = join_tracks_2(tracks, array_for_join)
    print_track(tracks, array_for_join, array_for_join_points, merged, before)

def join_tracks_2(tracks, tracks_for_merge):
    print("start joining")
    merge_tracks_array = []
    # create list with merge tracks
    for track_index in range(len(tracks)):
        for merge_index in range(len(tracks_for_merge[track_index])):
            merge = MergeTracks(tracks[track_index], tracks_for_merge[track_index][merge_index])
            merge.mean_squared_error()
            merge_tracks_array.append(merge)
    # print(merge_tracks_array)
    merge_tracks_array.sort(key=lambda merge_track : merge_track.sum)
    final_array = []
    while len(merge_tracks_array) > 0:
        first = merge_tracks_array[0].first_track
        second = merge_tracks_array[0].second_track
        first.merge_tracks(second)
        final_array.append(first)
        del merge_tracks_array[0]
        copy = merge_tracks_array.copy()
        index_for_del = []
        for index in range(len(copy)):
            if copy[index].first_track == first:
                index_for_del.append(index)
            if copy[index].first_track == second:
                merge_tracks_array[index].first_track = first
            if copy[index].second_track == second:
                index_for_del.append(index)
        for ind in reversed(index_for_del):
            del merge_tracks_array[ind]


    return final_array


def join_tracks(tracks, tracks_for_merge):
    print("start join")
    for index1 in range(len(tracks)):
        tracks[index1].compute_speed()
        #tracks[index1].compute_direction()
        tracks[index1].compute_avg_vector()
        tracks[index1].angle = angle(tracks[index1].vector)

    for index1 in range(len(tracks)):
        track_angle = tracks[index1].angle
        minimum_index = -1
        minimum = 10000000
        for index2 in range(len(tracks_for_merge[index1])):
            diff = abs(track_angle - tracks_for_merge[index1][index2].angle)
            if diff < minimum:
                minimum = diff
                minimum_index = index2
        # merge tracks
        if minimum_index != -1:
            tracks[index1].merge_tracks(tracks_for_merge[index1][minimum_index])
            # delete track
            track_for_delete = tracks_for_merge[index1][minimum_index]
            for del_index1 in range(len(tracks_for_merge)):
                for del_index2 in range(len(tracks_for_merge[del_index1])):
                    if track_for_delete == tracks_for_merge[del_index1][del_index2]:
                        del tracks_for_merge[del_index1][del_index2]
    return tracks,tracks_for_merge

def create_tracks(track_array):
    tracks = []
    for track in track_array:
        tracks.append(Track(track))
    return tracks


def create_bounding_boxes(bb_array):
    if bb_array is None:
        return None
    bounding_boxes = []
    for bb in bb_array:
        bounding_boxes.append(BoundingBox(bb[0], bb[1], bb[2], bb[3], 0, 0))
    return bounding_boxes


def is_in_radius(bb1, bb2, radius):
    return get_distance(bb1, bb2) <= radius


def get_distance(bb1, bb2):
    return math.sqrt((bb1.x - bb2.x)**2 + (bb1.y - bb2.y)**2)

def print_track(tracks, merging, merge_point, merged,before):
    # Create a black image
    img = np.zeros((720, 1280, 3), np.uint8)
    img2 = np.zeros((720, 1280, 3), np.uint8)
    img3 = np.zeros((720, 1280, 3), np.uint8)
    for track_index in range(len(tracks)):
        color = (random.randint(0,255), random.randint(0,255), random.randint(0,255))
        next_x = -1
        next_y = -1
        for index in range(len(tracks[track_index].bounding_boxes) - 1):
            x = tracks[track_index].bounding_boxes[index].x
            y = tracks[track_index].bounding_boxes[index].y
            next_x = tracks[track_index].bounding_boxes[index + 1].x
            next_y = tracks[track_index].bounding_boxes[index + 1].y
            cv2.line(img, (x, y), (next_x, next_y), color)
        '''for merge in merging[track_index]:
            if next_y == 1 or next_x == -1:
                break
            first_bb = merge.bounding_boxes[0]
            merge_x = first_bb.x
            merge_y = first_bb.y
            cv2.line(img, (next_x, next_y), (merge_x, merge_y), (0,0,255))
        for point in merge_point[track_index]:
            #cv2.line(img, (point.x, point.y), (point.x, point.y), (0, 255, 0))
            img[point.y][point.x] = (0, 255, 0)'''

    for track_index in range(len(before)):
        color = (random.randint(0,255), random.randint(0,255), random.randint(0,255))
        for index in range(len(before[track_index].bounding_boxes) - 1):
            x = before[track_index].bounding_boxes[index].x
            y = before[track_index].bounding_boxes[index].y
            next_x = before[track_index].bounding_boxes[index + 1].x
            next_y = before[track_index].bounding_boxes[index + 1].y
            cv2.line(img3, (x, y), (next_x, next_y), color)

    for track_index in range(len(merged)):
        color = (random.randint(0,255), random.randint(0,255), random.randint(0,255))
        for index in range(len(merged[track_index].bounding_boxes) - 1):
            x = merged[track_index].bounding_boxes[index].x
            y = merged[track_index].bounding_boxes[index].y
            next_x = merged[track_index].bounding_boxes[index + 1].x
            next_y = merged[track_index].bounding_boxes[index + 1].y
            cv2.line(img2, (x, y), (next_x, next_y), color)

    cv2.imshow("Pred", img3)
    #cv2.imshow("join", img)
    cv2.imshow("Po", img2)
    # If q is pressed then exit program
    k = cv2.waitKey(0)
    if k == ord('q'):
        cv2.destroyAllWindows()

# calculate vector
# (a1,a2,0)×(b1,b2,0)=(0,0,a1b2−a2b1)
def vector(v1 , v2):
    pass

def angle(v1):
    # tangens alfa = protilahla / prilahla = y / x
    print(v1)
    if v1[0] == 0 or v1[1] == 0:
        return 0
    tan_v1_alfa = abs(v1[1] / v1[0])
    #tan_v2_alfa = abs(v2[1] / v2[0])

    v1_alfa = math.atan(tan_v1_alfa)
    #v2_alfa = math.atan(tan_v2_alfa)
    return v1_alfa

