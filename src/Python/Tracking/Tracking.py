from Track import *
from BoundingBox import *
import math
import cv2
import numpy as np
import random
import datetime, os


def BoundingBoxmerge_tracks(track_array, unresolved_points, frame_difference, frameCount, radius=50):
    """
    Hlavna funkcia ktora sluzi na mergovanie tras. Na zaciatku sa prerobia trasy z maticneho tvaru na objekty.
    Nasledne sa najdu kandidati na spojenie podla parametra difference a radius. Potom sa trasy spoja na zaklade
    tokovej matice alebo vypoctu mean squared error.
    :param track_array: pole tras vo formate matice bodov kde kazdy bod je definovany [x, y, zaradeny, frame]
    :param unresolved_points: pole nezaradenych bodov
    :param frame_difference: maximalny pocet bynechanych framov
    :param radius: maximalne okolie ktore sa prehladava
    :return: pole pospajanych tras vo formate matice
    """
    sum0 = 0
    for tr in track_array:
        sum0 += len(tr)

    tracks = create_tracks(track_array)
    print_info(tracks, frameCount)
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

        # check tracks for merge
        for track_for_merge in tracks:
            if track != track_for_merge:
                # compare last point from first track with first point in second track
                track2_first_bb = track_for_merge.bounding_boxes[0]
                track1_last_bb = track.bounding_boxes[-1]
                if is_in_radius(track1_last_bb, track2_first_bb, radius) \
                        and track1_last_bb.frame_index < track2_first_bb.frame_index <= track1_last_bb.frame_index + frame_difference:
                    tracks_in_radius.append(Track(None, track_for_merge))

        # add array of all tracks which can be joined to track
        array_for_join.append(tracks_in_radius)
        # add array of all points which can be joined to track
        array_for_join_points.append(points_in_radius)

    merged = join_tracks_min_error(tracks, array_for_join, True, False)
    print_info(merged, frameCount)

    sum1 = 0
    sum2 = 0
    for tr in before:
        sum1 += len(tr.bounding_boxes)
    for tr in merged:
        sum2 += len(tr.bounding_boxes)

    print('sum0=' + str(sum0) + 'sum1=' + str(sum1) + 'sum2=' + str(sum2))
    merged = track_object_to_matrix(merged)
    return merged


def merge_tracks_flow_matrix(track_array, flow_matrix, frame_difference, radius=50):
    """
    Spojenie tras na zaklade tokovej matice. Na zaciatku sa najdu kandidati na spojenie podla parametra difference.
    Nasledne podla tokovej matice sa vypocita cesta z bodu [X1,Y1] na frame Z1 do bodu [X2,Y2] na frame Z2 tak, ze ku bodu X1,Y1 sa pripocita hodnota vektoru na tomto bode. Operacia sa opakuje
    (Z2 - Z1) krat.
    :param track_array: pole tras vo formate matice bodov kde kazdy bod je definovany [x, y, zaradeny, frame]
    :param flow_matrix: tokova matica
    :param frame_difference: maximalny rozdiel framov
    :param radius: maximalny radius
    :return: maticu spojenych tras
    """
    sum0 = 0
    for tr in track_array:
        sum0 += len(tr)
    tracks = create_tracks(track_array)

    print_info(tracks)
    before = []
    for track in tracks:
        before.append(Track(None, track))
    # loop all tracks
    array_for_join = []

    for track in tracks:
        tracks_in_frame_range = []
        # check tracks for merge
        for track_for_merge in tracks:
            if track != track_for_merge:
                # compare last point from first track with first point in second track
                track2_first_bb = track_for_merge.bounding_boxes[0].frame_index
                track1_last_bb = track.bounding_boxes[-1].frame_index
                if track1_last_bb < track2_first_bb <= track1_last_bb + frame_difference:
                    tracks_in_frame_range.append(track_for_merge)
                    # add array of all tracks which can be joined to track

        array_for_join.append(tracks_in_frame_range)

    new_tracks = join_tracks_flow_matrix(tracks, array_for_join, flow_matrix, radius)
    print_info(new_tracks)

    print('done')
    sum1 = 0
    sum2 = 0
    for tr in before:
        sum1 += len(tr.bounding_boxes)
    for tr in new_tracks:
        sum2 += len(tr.bounding_boxes)

    print('sum0=' + str(sum0) + 'sum1=' + str(sum1)+ 'sum2=' + str(sum2))
    new_tracks = track_object_to_matrix(new_tracks)
    return new_tracks


def join_tracks_flow_matrix(tracks, tracks_for_merge, flow_matrix, max_radius):
    """
    Funkcia spaja trasy podla tokovej matice. Pre kazdu trasu a jej kandidata vypocita vzdialenost podla tokovej matice.
    Ak je vzdialenost mensia ako paramater radius, prida sa do pola na spajanie spolu s informaciou o vzdialenosti.
    Nasledne sa toto pole utriedi podla vzdialenosti od najmensej a spoja sa trasy. Trasy ktore sa spoja su z pola zmazane
    podla nasledovnych pravidiel. Nech A je povodna trasa a B je jeho kandidat na spojenie.
    1. Zmazat vsetky prvky pola kde kandidat na spojenie je trasa B.
    2. Zmazat vsetky prvky pola kde povodna trasa je trasa A.
    3. Najst vsetky prvky pola kde kandidat na spojenie je trasa A a nahradit ho novou trasou AB.
    :param tracks: pole tras
    :param tracks_for_merge: pole kandidatov pre kazdu trasu z pola tracks
    :param flow_matrix: tokova matica
    :param max_radius: maximalna vzdialenost
    :return:
    """
    x_max = len(flow_matrix)
    y_max = len(flow_matrix[0])
    print('x max=' + str(x_max))
    print('y max=' + str(y_max))

    merge_tracks_array = []
    for track_index in range(len(tracks)):
        # cislo framu poslednej bunky v trase1
        frame = tracks[track_index].bounding_boxes[-1].frame_index
        # pre kazdeho kandidata vypocitat vzdialenost podla tokovej matice
        for merge_index in range(len(tracks_for_merge[track_index])):
            x = tracks[track_index].bounding_boxes[-1].x
            y = tracks[track_index].bounding_boxes[-1].y
            # prva bunka v trase ktora je kandidat na spojenie ku trase1
            second_track_frame_index = tracks_for_merge[track_index][merge_index].bounding_boxes[0].frame_index
            frame_diff = second_track_frame_index - frame
            # vypocitat na akej pozicii sa bude nachadzat bod podla tokovej matice
            for index in range(frame_diff):
                # pripocitat hodnotu podla tokovej matice
                x_int = int(x)
                y_int = int(y)
                if x_int < x_max and y_int < y_max:
                    x_temp = flow_matrix[x_int][y_int][1][0]
                    y_temp = flow_matrix[x_int][y_int][1][1]
                    angle_radiant = get_vector_angle(x_temp, y_temp)
                    x, y = get_position(x, y, angle_radiant, tracks[track_index].speed)
                    #print('x=' + str(x_int) + ' y=' + str(y_int))
                    #x += flow_matrix[x_int][y_int][1][0]
                    #y += flow_matrix[x_int][y_int][1][1]
            first_x = tracks_for_merge[track_index][merge_index].bounding_boxes[0].x
            first_y = tracks_for_merge[track_index][merge_index].bounding_boxes[0].y
            distance = get_distance_array([first_x, first_y], [x, y])
            if distance < max_radius:
                new_merge_track = MergeTracks(Track(None, tracks[track_index]), Track(None, tracks_for_merge[track_index][merge_index]))
                new_merge_track.distance = distance
                merge_tracks_array.append(new_merge_track)

    merge_tracks_array.sort(key=lambda merge_track: merge_track.distance)
    final_array = []
    # postupne pospajat trasy
    while len(merge_tracks_array) > 0:
        first = merge_tracks_array[0].first_track
        first_copy = Track(None, first)
        second = merge_tracks_array[0].second_track
        #todo del
        if first in tracks:
            tracks.remove(first)
        if second in tracks:
            tracks.remove(second)
        first.merge_tracks(second)
        # pridat spojenu trasu do vysledneho pola
        final_array.append(first)
        # zmazat spojenu trasu
        del merge_tracks_array[0]
        copy = merge_tracks_array.copy()
        # pole indexov na zmazanie
        index_for_del = []
        for index in range(len(copy)):
            if copy[index].first_track == first_copy:
                index_for_del.append(index)
            if copy[index].first_track == second:
                merge_tracks_array[index].first_track = first
            if copy[index].second_track == first_copy:
                merge_tracks_array[index].second_track = first
            if copy[index].second_track == second:
                index_for_del.append(index)
            if copy[index].first_track == first:
                index_for_del.append(index)
            if copy[index].second_track == first:
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
    # print_track2(no_duplicate_array,  no_duplicate_array)
    return no_duplicate_array


def join_tracks_min_error(tracks, tracks_for_merge, method1=True, method2=False):
    """
    Spoji trasy na zaklade najmensej chyby spojenia. Chyba spojenia sa pocita 3 funkciami.
    :param tracks: pole tras
    :param tracks_for_merge: pole kandidatov pre kazdu trasu
    :param method1: ak je True pouzije sa funkcia mean squared error z N poslednych bodov
    :param method2: ak je True pouzije sa funkcia mean squared error alfa n
    :return: spojene trasy
    """
    merge_tracks_array = []
    # create list with merge tracks
    for track_index in range(len(tracks)):
        for merge_index in range(len(tracks_for_merge[track_index])):
            merge = MergeTracks(Track(None, tracks[track_index]), Track(None, tracks_for_merge[track_index][merge_index]))
            if method1:
                merge.mean_squared_error_n_last(8)
            elif method2:
                merge.mean_squared_error_alfa_n()
            merge_tracks_array.append(merge)

    # zoradit pole podla chyby spojenia
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
            if copy[index].first_track == first_copy:
                index_for_del.append(index)
            if copy[index].first_track == second:
                merge_tracks_array[index].first_track = first
            if copy[index].second_track == first_copy:
                merge_tracks_array[index].second_track = first
            if copy[index].second_track == second:
                index_for_del.append(index)
            if copy[index].first_track == first:
                index_for_del.append(index)
            if copy[index].second_track == first:
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


def create_tracks(track_array):
    """
    Prerobi maticu tras na maticu objektov Track.
    :param track_array: pole tras
    :return: pole objektov
    """
    tracks = []
    track_id = 0
    for track in track_array:
        t = Track(track)
        t.id = track_id
        t.compute_speed()
        track_id += 1
        tracks.append(t)
    return tracks


def create_bounding_boxes(bb_array):
    """
    Prerobi maticu nezaradenych bodov na maticu objektov BoundingBox.
    :param bb_array: pole tras
    :return: pole objektov
    """
    if bb_array is None:
        return None
    bounding_boxes = []
    for bb in bb_array:
        bounding_boxes.append(TrackBoundingBox(bb[0], bb[1], bb[2], bb[3], 0, 0))
    return bounding_boxes


def is_in_radius(bb1, bb2, radius):
    return get_distance_object(bb1, bb2) <= radius


def get_distance_object(bb1, bb2):
    """
    Vrati vzdialenost medzi dvoma objektami BoundingBox.
    :param bb1: bounding box a
    :param bb2: bounding box b
    :return: vzdialenost
    """
    distance = math.sqrt((bb1.x - bb2.x)**2 + (bb1.y - bb2.y)**2)
    return distance


def get_distance_array(a, b):
    """
    Vrati vzdialenost medzi dvoma bodmi, kde bod je pole [X, Y]
    :param a: bod a
    :param b: bod b
    :return: vzdialenost
    """
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
    """
    Vypocita uhol vectoru.
    :param x: bod x
    :param y: bod y
    :return: uhol
    """
    vect = math.atan2(y, x)
    return vect


def get_position(x, y, angle, speed):
    """
    Vypocita poziciu bodu podla uhlu a rychlosti.
    X=distance*cos(angle) + x0
    Y=distance*sin(angle) + y0
    :param x: povodna x-ova pozicia
    :param y: povodna y-ova pozicia
    :param angle: uhol
    :param speed: rychlost
    :return: novu poziciu [x, y]
    """
    x_new = x + math.cos(angle)*speed
    y_new = y + math.cos(angle)*speed
    return x_new, y_new


def print_info(tracks, frameCount):
    #snimok
    #pocet tras
    #priemerna dlzka trasy
    print('Snimok=' + str(frameCount))
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
    # TODO pomocna funkcia na testovanie
    new_array = []

    for track_index in range(len(tracks)):
        max = -1
        max_index = -1
        t = -1
        for index in range(len(tracks[track_index].bounding_boxes) - 1):
            track1 = tracks[track_index].bounding_boxes[index]
            track2 = tracks[track_index].bounding_boxes[index + 1]
            distance = get_distance_object(track1, track2)
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
    """
    Funkcia prerobi pole objektov Track na pole trackov.
    :param tracks: pole objektov Track
    :return: pole tras
    """
    new_tracks = []
    for track in tracks:
        new_track = []
        for bb in track.bounding_boxes:
            new_bb = [bb.x,bb.y,bb.in_track,bb.frame_index,bb.width,bb.height]
            new_track.append(new_bb)
        new_tracks.append(new_track)

    return new_tracks


def compare_tracks(file_ground_truth, flow_matrix):
    import XMLParser
    tracks_gt, mat, src_name = XMLParser.parseXMLData(file_ground_truth)
    result = merge_tracks_flow_matrix(tracks_gt, flow_matrix, 5)
    print(result)
    '''tracks_gt = create_tracks(tracks_gt)
    for t in tracks_gt:
        for bb in t.bounding_boxes:
            if bb.x == 858 and bb.y == 200:
                print('found')
    print('daco')'''

