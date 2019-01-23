import XMLParser, Track


def load_tracks(file_ground_truth, file_alg):
    print('loading')
    tracks_gt, mat, src_name = XMLParser.parse_xml_anastroj_v2(file_ground_truth)
    tracks_alg, mat, src_name = XMLParser.parse_xml_anastroj(file_alg)
    print('loaded')
    tracks_objects_gt = Track.create_tracks(tracks_gt)
    tracks_objects_alg = Track.create_tracks(tracks_alg)
    print('created')
    result = true_positive_segments(tracks_objects_gt, tracks_objects_alg)
    result2 = get_true_positive_segments(tracks_objects_gt, tracks_objects_alg)
    print('true segments')
    #areas = bad_segments_area(tracks_objects_alg, tracks_objects_gt, 1280, 720)
    count = get_count_of_segments(tracks_objects_gt)
    count2 = get_count_of_segments(tracks_objects_alg)
    print('count GT segments=' + str(count))
    print('count ALG segments=' + str(count2))
    #false_p = false_positive_segments(tracks_objects_alg, result2)
    #false_n = false_negative_segments(tracks_objects_gt, result2)
    false_p2 = false_posisive_segments2(tracks_objects_alg)
    false_n2 = false_negative_segments2(tracks_objects_gt)
    print(result)
    print(result2)
    #print('false p=' + str(false_p))
    #print('false n=' + str(false_n))
    print('false p=' + str(false_p2))
    print('false n=' + str(false_n2))


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


def full_tracks(tracks_gt, tracks_alg):
    for track_alg in tracks_alg:
        for track_gt in tracks_gt:
            found = False
            for bb_gt in track_gt.bounding_boxes:
                # porovnat s prvym bodom v trase a najst tak GROUND TRUTH pre kazdu trau ktoru nasiel algoritmus
                if bb_gt == track_alg.bounding_boxes[0]:
                    track_alg.ground_truth = track_gt
                    found = True
                    break
            if found:
                break

    full_tracks_count = 0
    for track_alg in tracks_alg:
        is_full = True
        if len(track_alg.bounding_boxes) == len(track_alg.ground_truth.bounding_boxes):
            for index,bb in enumerate(track_alg.bounding_boxes):
                if track_alg.bounding_boxes[index] != track_alg.ground_truth.bounding_boxes[index]:
                    is_full = False
                    break
            if is_full:
                full_tracks_count += 1

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


def get_true_positive_segments(tracks_gt, tracks_alg):
    count = 0
    count2 = 0
    count3 = 0
    for track_gt in tracks_gt:
        for bb_gt in range(len(track_gt.bounding_boxes) - 1):
            found = False
            for track_alg in tracks_alg:
                for bb_alg in range(len(track_alg.bounding_boxes) - 1):
                    if track_gt.bounding_boxes[bb_gt] == track_alg.bounding_boxes[bb_alg] and \
                                    track_gt.bounding_boxes[bb_gt + 1] == track_alg.bounding_boxes[bb_alg + 1]:
                        count += 1
                        track_gt.bounding_boxes[bb_gt].true_positive = True
                        #track_gt.bounding_boxes[bb_gt + 1].true_positive = True
                        track_alg.bounding_boxes[bb_alg].true_positive = True
                        #track_alg.bounding_boxes[bb_alg + 1].true_positive = True
                        found = True
                        break
                    frame_diff = track_alg.bounding_boxes[bb_alg + 1].frame_index - track_alg.bounding_boxes[bb_alg].frame_index
                    if frame_diff > 1 and track_gt.bounding_boxes[bb_gt] == track_alg.bounding_boxes[bb_alg] and \
                                    track_gt.bounding_boxes[bb_gt + frame_diff] == track_alg.bounding_boxes[bb_alg + 1]:
                        track_alg.bounding_boxes[bb_alg].true_positive2 = True
                        count2 += 1
                    elif frame_diff > 1 and track_gt.bounding_boxes[bb_gt] == track_alg.bounding_boxes[bb_alg]\
                            and track_gt.bounding_boxes[bb_gt + frame_diff] != track_alg.bounding_boxes[bb_alg + 1]:
                        count3 += 1
                if found:
                    break
    print('Dobre spojene=' + str(count2))
    print('Zle spojenie=' + str(count3))
    return count


def get_count_of_segments(tracks):
    count = 0
    for track in tracks:
        count += len(track.bounding_boxes) - 1

    return count

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

def false_negative_segments2(tracks_gt):
    segments = 0
    for track in tracks_gt:
        for bb_ind in range(len(track.bounding_boxes) - 1):
            if not track.bounding_boxes[bb_ind].true_positive:
                segments += 1
    return segments

def false_posisive_segments2(tracks_alg):
    segments = 0
    for track in tracks_alg:
        for bb_ind in range(len(track.bounding_boxes) - 1):
            if not track.bounding_boxes[bb_ind].true_positive and not track.bounding_boxes[bb_ind].true_positive2:
                segments += 1
    return segments

def bad_segments_area(tracks_alg, tracks_gt, width, height, radius=30):
    cols = int(width / radius)
    rows = int(height / radius)
    array_fp = [[0 for x in range(cols)] for y in range(rows)]
    array_fn = [[0 for x in range(cols)] for y in range(rows)]



    for track in tracks_alg:
        for bb in track.bounding_boxes:
            if not bb.true_positive:
                #print('bbx=' + str(bb.x) + ' bby=' + str(bb.y))
                x_index = int(bb.x / radius)
                y_index = int(bb.y / radius)
                #print('x=' + str(x_index) + ' y=' + str(y_index))
                array_fn[y_index][x_index] += 1

    for track in tracks_gt:
        for bb in track.bounding_boxes:
            if not bb.true_positive:
                x_index = int(bb.x / radius)
                y_index = int(bb.y / radius)
                array_fp[y_index][x_index] += 1

    print(array_fp)
    print(array_fn)


def save_test():
    mat = [[[0,0,0,0,0,0]],[[1,1,1,1,1,1]],[[2,2,2,2,2,2]], [[3,3,3,3,3,3]]]
    tracks = [[[0,0,0,0,0,0], [1,1,1,1,1,1], [2,2,2,2,2,2]]]
    src = ['a', 'b', 'c','d','e','f','g']
    XMLParser.save_as_anastroj_file(mat, tracks, src, 'daco.xml')


def compare_tracks(file_ground_truth, file_alg):
    tracks_gt, mat, src_name = XMLParser.parse_xml_anastroj2(file_ground_truth)
    tracks_alg, mat, src_name = XMLParser.parse_xml_anastroj2(file_alg)

    tracks_objects_gt = Track.create_tracks(tracks_gt)
    tracks_objects_alg = Track.create_tracks(tracks_alg)
    not_found = 0
    sum1 = 0
    sum2 = 0
    for tr in tracks_objects_gt:
        sum1 += len(tr.bounding_boxes)
    for tr in tracks_objects_alg:
        sum2 += len(tr.bounding_boxes)

    print('sum1=' + str(sum1) + 'sum2=' + str(sum2))
    for track in tracks_objects_gt:
        for bb in track.bounding_boxes:
            found = False
            for track2 in tracks_objects_alg:
                for bb2 in track2.bounding_boxes:
                    if bb == bb2:
                        found = True
                        break
            if found:
                break
            else:
                not_found += 1
                print('NOT FOUND Bounding box=' + str(bb) + ' in track=' + str(track.id))

    print('Total=' + str(not_found))
# flow_matrix_merge_200 .... julia_tracking_200_frame
# load_tracks("C:\\Users\\Miroslav Buzgo\\Desktop\\21-1-merge.xml", "C:\\Users\\Miroslav Buzgo\\Desktop\\tracks_1_200.xml")
load_tracks("C:\\Users\\Miroslav Buzgo\\Desktop\\tracks_1_200.xml", "C:\\Users\\Miroslav Buzgo\\Desktop\\merge-20-5-random.xml")
# load_tracks("C:\\Users\\Miroslav Buzgo\\Desktop\\tracks_1_200.xml", "C:\\Users\\Miroslav Buzgo\\Desktop\\julia-12-8-8-random.xml")
#load_tracks("C:\\Users\\Miroslav Buzgo\\Desktop\\tracks_1_200.xml", "C:\\Users\\Miroslav Buzgo\\Desktop\\flow-matrix-20-5-random.xml")
# save_test()
# compare_tracks("C:\\Users\\Miroslav Buzgo\\Desktop\\julia200.xml", "C:\\Users\\Miroslav Buzgo\\Desktop\\merge200.xml")
