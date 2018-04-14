from Track import *
from BoundingBox import *
import math
import cv2
import numpy as np
import random
import datetime, os
import sys
import Test


def merge_tracks(track_array, unresolved_points, difference, radius=50):
    tracks = create_tracks(track_array)
    print_info(tracks)
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
        #print('track=' + str(track))
        for index in range(difference):
            #porovnaju sa body
            new_x = (2 + index) * track.bounding_boxes[-1].x - (index + 1) * track.bounding_boxes[-2].x
            new_y = (2 + index) * track.bounding_boxes[-1].y - (index + 1) * track.bounding_boxes[-2].y
            #print(str(new_x) + ' , ' + str(new_y))
            frame_index = track.bounding_boxes[-1].frame_index + (index + 1)  # start from 0
            new_bb = BoundingBox(new_x, new_y, 1, frame_index, 0, 0)
            # check tracks for merge
            for track_for_merge in tracks:
                if track_for_merge.bounding_boxes[0].frame_index == 15:
                    c = 4
                if track != track_for_merge:
                    # compare last point from first track with first point in second track
                    track2_first_p = track_for_merge.bounding_boxes[0]
                    if is_in_radius(new_bb, track2_first_p, radius) and new_bb.frame_index == track2_first_p.frame_index:
                        tracks_in_radius.append(Track(None,track_for_merge))
            # check unresolved points for merge
            for point in unresolved_points:
                if is_in_radius(new_bb, point, radius) and new_bb.frame_index == point.frame_index:
                    points_in_radius.append(point)
        # add array of all tracks which can be joined to track
        array_for_join.append(tracks_in_radius)
        # add array of all points which can be joined to track
        array_for_join_points.append(points_in_radius)

    '''before = []
    for track_index in range(len(tracks)):
        if tracks[track_index].id == 39:
            before.append(Track(None,tracks[track_index]))
        for merge_index in range(len(array_for_join[track_index])):
            if tracks[track_index].id == 39:
                before.append(Track(None,array_for_join[track_index][merge_index]))'''

    merged = join_tracks_2(tracks, array_for_join, True, False, False)
    print_info(merged)
    #merged = get_longest_track(merged)

    print_track2(merged, before)
    merged = track_object_to_matrix(merged)
    return merged

def merge_tracks_flow_matrix(track_array, flow_matrix, difference, radius = 50):
    tracks = create_tracks(track_array)
    print_info(tracks)
    before = []
    for track in tracks:
        before.append(Track(None, track))
    # loop all tracks
    array_for_join = []
    for track in tracks:
        # for all 1 to frame difference add point and check tracks for merge
        tracks_in_radius = []
        for index in range(difference):
            # porovnaju sa body
            frame_index = track.bounding_boxes[-1].frame_index + (index + 1)
            # check tracks for merge
            for track_for_merge in tracks:
                if track != track_for_merge:
                    # compare last point from first track with first point in second track
                    track2_first_p = track_for_merge.bounding_boxes[0]
                    if frame_index == track2_first_p.frame_index:
                        tracks_in_radius.append(track_for_merge)
        # add array of all tracks which can be joined to track
        array_for_join.append(tracks_in_radius)
    new_tracks = join_tracks_flow_matrix(tracks,array_for_join,flow_matrix, radius)
    print_info(new_tracks)
    #new_tracks = get_longest_track(new_tracks)
    print_track2(new_tracks, before)
    new_tracks = track_object_to_matrix(new_tracks)
    '''print('----')
    for track_index in range(len(new_tracks)):
        pom = ''
        for bb in new_tracks[track_index].bounding_boxes:
            pom += str(bb.frame_index) + ','
        print(pom)

    print('-----')'''
    return new_tracks

def join_tracks_flow_matrix(tracks, tracks_for_merge, flow_matrix,max_radius):
    x_max = len(flow_matrix)
    y_max = len(flow_matrix[0])
    print('x max=' + str(x_max))
    print('y max=' + str(y_max))

    merge_tracks_array = []
    for track_index in range(len(tracks)):
        if tracks[track_index].id == 213:
            print('error')
        best_ind = -1
        best_val = sys.maxsize
        # cislo poslednej bunky v tracku
        frame = tracks[track_index].bounding_boxes[-1].frame_index
        for merge_index in range(len(tracks_for_merge[track_index])):
            x = tracks[track_index].bounding_boxes[-1].x
            y = tracks[track_index].bounding_boxes[-1].y
            #prva bunka v trase
            second_track_frame_index = tracks_for_merge[track_index][merge_index].bounding_boxes[0].frame_index
            frame_diff =  second_track_frame_index - frame
            for index in range(frame_diff):
                # pripocitat hodnotu podla tokovej matice
                x_int = int(x)
                y_int = int(y)
                if x_int < x_max and y_int < y_max:
                    x_temp = flow_matrix[x_int][y_int][1][0]
                    y_temp = flow_matrix[x_int][y_int][1][1]
                    angle_radiant = get_vector_angle(x_temp, y_temp)
                    x, y = get_position(x, y, angle_radiant, tracks[track_index].speed)
            first_x = tracks_for_merge[track_index][merge_index].bounding_boxes[0].x
            first_y = tracks_for_merge[track_index][merge_index].bounding_boxes[0].y
            distance = get_distance2([first_x, first_y],[x,y])
            if distance < max_radius:
                new_merge_track = MergeTracks(Track(None,tracks[track_index]),Track(None,tracks_for_merge[track_index][merge_index]))
                new_merge_track.distance = distance
                merge_tracks_array.append(new_merge_track)
            '''if distance < best_val and distance < max_radius:
                best_ind = merge_index
                best_val = distance
        if best_ind != -1:
            print('merge')
            track = tracks_for_merge[track_index][best_ind]
            copy = Track(None, track)
            tracks[track_index].merge_tracks(tracks_for_merge[track_index][best_ind])
            new_tracks.append(tracks[track_index])
            for t1 in range(len(tracks)):
                if copy in tracks_for_merge[t1]:
                    tracks_for_merge[t1].remove(copy)'''
    merge_tracks_array.sort(key=lambda merge_track : merge_track.distance)
    final_array = []
    while len(merge_tracks_array) > 0:
        first = merge_tracks_array[0].first_track
        first_copy = Track(None, first)
        second = merge_tracks_array[0].second_track
        if first in tracks:
            tracks.remove(first)
        if second in tracks:
            tracks.remove(second)
        first.merge_tracks(second)
        final_array.append(first)
        del merge_tracks_array[0]
        copy = merge_tracks_array.copy()
        index_for_del = []
        for index in range(len(copy)):
            if copy[index].first_track == first:
                index_for_del.append(index)
            if copy[index].second_track == first:
                index_for_del.append(index)
            if copy[index].first_track == first_copy:
                index_for_del.append(index)
            if copy[index].second_track == first_copy:
                index_for_del.append(index)
            if copy[index].first_track == second:
                merge_tracks_array[index].first_track = first
            if copy[index].second_track == second:
                index_for_del.append(index)
        for ind in reversed(index_for_del):
            del merge_tracks_array[ind]

    no_duplicate_array = []

    for track in final_array:
        if track not in no_duplicate_array:
            no_duplicate_array.append(track)
    for track in tracks:
        if track not in no_duplicate_array:
            no_duplicate_array.append(track)

    return no_duplicate_array

def join_tracks_2(tracks, tracks_for_merge, method1 = True, method2 = False, method3 = False):
    merge_tracks_array = []
    # create list with merge tracks
    for track_index in range(len(tracks)):
        for merge_index in range(len(tracks_for_merge[track_index])):
            if tracks[track_index].id == 39 or tracks_for_merge[track_index][merge_index].id == 39:
                a = 0
            merge = MergeTracks(Track(None,tracks[track_index]), Track(None,tracks_for_merge[track_index][merge_index]))
            if method1:
                #merge.mean_squared_error()
                merge.mean_squared_error_n_last(8)
            elif method2:
                merge.mean_squared_error_alfa_n()
            elif method3:
                merge.mean_squared_error_alfa_nk(5)
            merge_tracks_array.append(merge)
    # print(merge_tracks_array)
    merge_tracks_array.sort(key=lambda merge_track : merge_track.sum)
    string = ''
    for tr in merge_tracks_array:
        string += str(tr.sum) + ','
    print(string)
    final_array = []
    while len(merge_tracks_array) > 0:
        first = merge_tracks_array[0].first_track
        first_copy = Track(None, first)
        second = merge_tracks_array[0].second_track
        if first in tracks:
            tracks.remove(first)
        if second in tracks:
            tracks.remove(second)
        first.merge_tracks(second)
        final_array.append(first)
        del merge_tracks_array[0]
        copy = merge_tracks_array.copy()
        index_for_del = []
        for index in range(len(copy)):
            if copy[index].first_track == first:
                index_for_del.append(index)
            if copy[index].second_track == first:
                index_for_del.append(index)
            if copy[index].first_track == first_copy:
                index_for_del.append(index)
            if copy[index].second_track == first_copy:
                index_for_del.append(index)
            if copy[index].first_track == second:
                merge_tracks_array[index].first_track = first
            if copy[index].second_track == second:
                index_for_del.append(index)
        for ind in reversed(index_for_del):
            del merge_tracks_array[ind]

    no_duplicate_array = []

    for track in final_array:
        if track not in no_duplicate_array:
            no_duplicate_array.append(track)
    for track in tracks:
        if track not in no_duplicate_array:
            no_duplicate_array.append(track)

    return no_duplicate_array


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
    id = 1
    for track in track_array:
        t = Track(track)
        t.id = id
        t.compute_speed()
        #print('track id=' + str(t.id) + ' speed=' + str(t.speed))
        id += 1
        tracks.append(t)
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
    distance = math.sqrt((bb1.x - bb2.x)**2 + (bb1.y - bb2.y)**2)
    return distance

def get_distance2(a, b):
    return math.sqrt((a[0] - b[0])**2 + (a[1] - b[1])**2)


def print_track2(tracks,old):
    # Create a black image
    img = np.zeros((720, 1280, 3), np.uint8)
    img2 = np.zeros((720, 1280, 3), np.uint8)
    print('len old=' + str(len(old)))
    print('len new=' + str(len(tracks)))
    for track_index in range(len(tracks)):
        color = (random.randint(0,255), random.randint(0,255), random.randint(0,255))
        for index in range(len(tracks[track_index].bounding_boxes) - 1):
            x = tracks[track_index].bounding_boxes[index].x
            y = tracks[track_index].bounding_boxes[index].y
            next_x = tracks[track_index].bounding_boxes[index + 1].x
            next_y = tracks[track_index].bounding_boxes[index + 1].y
            cv2.line(img, (x, y), (next_x, next_y), color)
    for track_index in range(len(old)):
        color = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
        for index in range(len(old[track_index].bounding_boxes) - 1):
            x = old[track_index].bounding_boxes[index].x
            y = old[track_index].bounding_boxes[index].y
            next_x = old[track_index].bounding_boxes[index + 1].x
            next_y = old[track_index].bounding_boxes[index + 1].y
            cv2.line(img2, (x, y), (next_x, next_y), color)

    # If q is pressed then exit program
    cv2.imwrite("after2.png",img)
    cv2.imshow("PO", img)
    cv2.imshow("PRED", img2)
    k = cv2.waitKey(0)
    if k == ord('q'):
        cv2.destroyAllWindows()


def print_track(tracks, merging, merge_point, merged, before, method, seed,file, num = -1):
    # Create a black image
    img = np.zeros((720, 1280, 3), np.uint8)
    img2 = np.zeros((720, 1280, 3), np.uint8)
    img3 = np.zeros((720, 1280, 3), np.uint8)
    img4 = np.zeros((720, 1280, 3), np.uint8)
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
        color = (random.randint(0, 255), random.randint(0,255), random.randint(0,255))
        color = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
        x = before[track_index].bounding_boxes[0].x
        y = before[track_index].bounding_boxes[0].y
        #cv2.putText(img3, "track id:" + str(track_index), (x, y - 5), cv2.FONT_ITALIC, 0.35, (0, 0, 255))
        for index in range(len(before[track_index].bounding_boxes) - 1):
            x = before[track_index].bounding_boxes[index].x
            y = before[track_index].bounding_boxes[index].y
            next_x = before[track_index].bounding_boxes[index + 1].x
            next_y = before[track_index].bounding_boxes[index + 1].y
            cv2.line(img3, (x, y), (next_x, next_y), color)

    for track_index in range(len(merged)):
        for index in range(len(merged[track_index].bounding_boxes) - 1):
            x = merged[track_index].bounding_boxes[index].x
            y = merged[track_index].bounding_boxes[index].y
            next_x = merged[track_index].bounding_boxes[index + 1].x
            next_y = merged[track_index].bounding_boxes[index + 1].y
            cv2.line(img2, (x, y), (next_x, next_y), color)

    if num != -1:
        for index in range(len(tracks[num].bounding_boxes) - 1):
            x = tracks[num].bounding_boxes[index].x
            y = tracks[num].bounding_boxes[index].y
            next_x = tracks[num].bounding_boxes[index + 1].x
            next_y = tracks[num].bounding_boxes[index + 1].y
            cv2.line(img4, (x, y), (next_x, next_y), (255, 0, 0))
    #cv2.imshow("1 track", img4)
    date = str(datetime.datetime.now().strftime("%Y-%m-%d-%H-%M"))
    dir_name = "test\\" + date + '_seed_'+str(seed)
    if not os.path.exists(dir_name):
        os.makedirs(dir_name)
        #cv2.imshow("Pred", img3)
    cv2.imwrite(dir_name + os.sep + file + "_" + method + "_file_pred.jpg", img3)
    #cv2.imshow("join", img)
    #cv2.imshow("Po", img2)
    cv2.imwrite(dir_name + os.sep + file + "_" + method + "_file_po.jpg", img2)
    # If q is pressed then exit program
    #k = cv2.waitKey(0)
    #if k == ord('q'):
    #    cv2.destroyAllWindows()


def get_vector_angle(x, y):
    vect = math.atan2(y, x)
    return vect

'''
X=distance*cos(angle) +x0
Y=distance*sin(angle) +y0
'''
def get_position(x, y, angle, speed):
    x_new = x + math.cos(angle)*speed
    y_new = y + math.cos(angle)*speed
    print('old x=' + str(x) + ' old y=' + str(y) + ' speed=' + str(speed) + ' angle=' + str(angle) + ' n x=' + str(x_new) + ' y new=' + str(y_new))
    return x_new, y_new


def print_info(tracks):
    #snimok
    #pocet tras
    #priemerna dlzka trasy
    print('Snimok=250')
    pocet_tras = len(tracks)
    print('Pocet tras=' + str(pocet_tras))
    sum_len = 0
    for track in tracks:
        sum_len += len(track.bounding_boxes)
    avg = sum_len/pocet_tras
    print('Avarage track=' + str(avg))

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

def get_longest_track(tracks):
    '''max = -1
    max_index = -1
    t = -1'''
    new_array = []

    for track_index in range(len(tracks)):
        max = -1
        max_index = -1
        t = -1
        for index in range(len(tracks[track_index].bounding_boxes) - 1):
            track1 = tracks[track_index].bounding_boxes[index]
            track2 = tracks[track_index].bounding_boxes[index + 1]
            distance = get_distance(track1, track2)
            if distance > max:
                max = distance
                max_index = index
                t = track_index
        if t != -1:
            new_array.append([max, tracks[t]])
    '''if t != -1:
        print('max=' + str(max))
        print('max index=' + str(max_index))
        print('track=' + str(tracks[t].id))
        print('track=' + str(tracks[t].bounding_boxes[max_index]) + ' -> ' + str(tracks[t].bounding_boxes[max_index + 1]))'''
    new_array.sort(key=lambda track: track[0])
    #return [tracks[t]]
    length = len(new_array)
    print(new_array[length - 1][1].id)
    print(new_array[length - 2][1].id)
    print(new_array[length - 3][1].id)
    print(new_array[length - 4][1].id)
    return [new_array[length - 1][1],new_array[length - 2][1],new_array[length - 3][1],new_array[length - 4][1],new_array[length - 5][1],new_array[length - 6][1],new_array[length - 7][1]]


def track_object_to_matrix(tracks):
    new_tracks = []
    for track in tracks:
        new_track = []
        for bb in track.bounding_boxes:
            new_bb = [bb.x,bb.y,bb.in_track,bb.frame_index,bb.width,bb.height]
            new_track.append(new_bb)
        new_tracks.append(new_track)

    return new_tracks

def test_tracks(gt_tracks, alg_tracks):
    gt_tracks2 = create_tracks(gt_tracks)
    alg_tracks2 = create_tracks(alg_tracks)
    Test.tracks_test(gt_tracks2, alg_tracks2)
