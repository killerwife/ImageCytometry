from Tracking import *
import random


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
