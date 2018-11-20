from Tracking import *
import random
import XMLParser, Track


def load_tracks(file_ground_truth, file_alg):
    tracks_gt, mat, src_name = XMLParser.parse_xml_anastroj2(file_ground_truth)
    tracks_alg, mat, src_name = XMLParser.parse_xml_anastroj2(file_alg)

    tracks_objects_gt = Track.create_tracks(tracks_gt)
    tracks_objects_alg = Track.create_tracks(tracks_alg)

    result = true_positive_segments(tracks_objects_gt, tracks_objects_alg)

    for track_alg in tracks_objects_alg:
        for track_gt in tracks_objects_gt:
            found = False
            for bb_gt in track_gt.bounding_boxes:
                # porovnat s prvym bodom v trase a najst tak GROUND TRUTH pre kazdu trau ktoru nasiel algoritmus
                if bb_gt == track_alg.bounding_boxes[0]:
                    track_alg.ground_truth = track_gt
                    found = True
                    break
            if found:
                break

    for track_alg in tracks_objects_alg:
        print("track_alg")
        track_alg.testing()

    print("Done.")



def tracks_test(ground_true_tracks, alg_tracks):
    matched_tracks = []
    # pre kazdy track z ground truth sa najde track z algoritmu kde ma najvacsi pocet rovnakych bodov
    for track in ground_true_tracks:
        max_true = -1
        max_true_track = None
        for track2 in alg_tracks:
            true_bb = 0
            index = 0
            for bb1 in track.bounding_boxes:
                for bb2 in track2.bounding_boxes:
                    if bb1 == bb2:
                        true_bb += 1
            if true_bb > max_true:
                max_true = true_bb
                max_true_track = track2
        matched_tracks.append((track, max_true_track))

    # porovna dlzku
    for matched_track in matched_tracks:
        gt = matched_track[0]
        alg = matched_track[1]
        if len(gt.bounding_boxes) == len(alg.bounding_boxes):
            print('TRUE ground truth track length = alg track length , gt=' + str(len(gt.bounding_boxes)) + ' alg='
                  + str(len(alg.bounding_boxes)))
        else:
            print('FALSE ground truth track length != alg track length , gt=' + str(len(gt.bounding_boxes)) + ' alg='
                  + str(len(alg.bounding_boxes)))
        # porovnanie false negatives, body ktore  mali byt v trase ale niesu
        count = 0
        for bb1 in gt.bounding_boxes:
            for bb2 in alg.bounding_boxes:
                if bb1 == bb2:
                    count += 1
        gt_count = len(gt.bounding_boxes)
        alg_count = len(alg.bounding_boxes)
        false_negative = gt_count - count
        false_positive = alg_count - count
        print('track with id=' + str(gt.id) + ' false positive=' + str(false_positive) + ' false negative=' + str(
            false_negative))


def complete_tracks(tracks_gt, tracks_alg):
    count_completed_tracks = 0
    """for track_gt in tracks_gt:
        track_id_alg = 0
        for index_gt in range(len(track_gt.bounding_boxes)):
            for track_alg_index in range(len(tracks_alg)):
                count_cell_alg = 0
                track_id_alg = track_alg_index
                for index_alg in range(len(track_alg_index.bounding_boxes)): ## ???
                    #if ()"""
    pass

# uhadnute segmenty
def true_positive_segments(tracks_gt, tracks_alg):
    count_tp = 0
    count_fp = 0
    count_fn = 0
    fn_list = []
    fp_list = []
    for track_gt in tracks_gt:
        for index_gt in range(len(track_gt.bounding_boxes) - 1):
            found = False
            for track_alg in tracks_alg:
                for index_alg, bb_alg in enumerate(track_alg.bounding_boxes):
                    if index_alg != len(track_alg.bounding_boxes) - 1:
                        if index_gt == len(track_gt.bounding_boxes) - 1:
                            count_fp += 1
                            fp_list.append(track_gt.bounding_boxes[index_gt])
                        elif track_gt.bounding_boxes[index_gt] == track_alg.bounding_boxes[index_alg]:
                            found = True
                            if track_gt.bounding_boxes[index_gt + 1] == track_alg.bounding_boxes[index_alg + 1]:
                                count_tp += 1
                            else:
                                count_fp += 1
                                fp_list.append(track_gt.bounding_boxes[index_gt])

                            break
                    elif track_gt.bounding_boxes[index_gt] == track_alg.bounding_boxes[index_alg]:
                        count_fn += 1
                        fn_list.append(track_gt.bounding_boxes[index_gt])


                if found:
                    break
            if not found:
                count_fn += 1
                fn_list.append(track_gt.bounding_boxes[index_gt])

    return count_tp, count_fn, count_fp, fn_list, fp_list


# neuhadnute segmenty TODO
def false_negative_segments(tracks_gt, count_tp):
    segments = 0
    for track_gt in tracks_gt:
        segments += len(track_gt.bounding_boxes) - 1
    return segments - count_tp

# zle uhadnute segmenty TODO
def false_positive_segments(tracks_alg, count_tp):
    segments = 0
    for track_alg in tracks_alg:
        segments += len(track_alg.bounding_boxes) - 1
    return segments - count_tp

def bad_segments_area(tracks_alg, tracks_gt):
    pass

#load_tracks("C:\\Users\\Miroslav Buzgo\\Desktop\\tracks_1_200.xml", "C:\\Users\\Miroslav Buzgo\\Desktop\\tracks_1_200.xml")
#load_tracks("C:\\Users\\Miroslav Buzgo\\Desktop\\xml1_1.xml", "C:\\Users\\Miroslav Buzgo\\Desktop\\xml1_2.xml")
load_tracks("C:\\Users\\Janka\\Desktop\\xml1_1.xml", "C:\\Users\\Janka\\Desktop\\xml1_2.xml")