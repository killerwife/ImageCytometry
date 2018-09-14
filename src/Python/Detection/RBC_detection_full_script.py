import cv2
import numpy as np
from time import sleep
import xml.etree.cElementTree as ET
import imutils
from imutils import contours

import Detection_by_tracking

# ------------------------------------------------------------------------------

def run(path_source_folder, image_name_pattern, radius_of_circle, file_name_of_XML, frame_from, frame_to):
    print("-------------------------------------------------------------------------------")
    print("--------------------------- Red Blood Cells detection -------------------------")
    print("-------------------------------------------------------------------------------")
    print("		Source folder   		:	", path_source_folder)
    print("		Name pattern    		:	", image_name_pattern)
    print("		Export XML      		:	", file_name_of_XML)
    print("		Frames (from-to)		:	", frame_from, "-", frame_to)
    print("		Radius of circle 		:	", radius_of_circle)
    print("-------------------------------------------------------------------------------")
    print("                            Hough Transform - Circles                          ")
    print("-------------------------------------------------------------------------------")

    detection = []
    detection = Detection_by_tracking._parse_from_XML("predikcia_06_04.xml")

    RBC_detection_circle(path_source_folder, image_name_pattern, radius_of_circle, file_name_of_XML, frame_from, frame_to, detection)


    print("-------------------------------------------------------------------------------")


# ------------------------------------------------------------------------------
def RBC_detection_circle(path_source_folder, image_name_pattern, radius_of_object, file_name_of_XML, frame_from,
                         frame_to, detection):
    list_of_frames = []
    list_of_count_particles = []
    background_image = _get_background(path_source_folder, image_name_pattern, frame_from, frame_to)
    i = 1
    total = frame_to - frame_from + 1
    while  frame_from <= frame_to:
        src_image = cv2.imread(path_source_folder + "/" + image_name_pattern % frame_from, 0)

        bgr_sub_image = _background_subtraction(src_image, background_image)
        cv2.imwrite("BackgroudSubtraction/" + image_name_pattern % frame_from, bgr_sub_image) #°°

        #edges_image = _cany_edge_detection(bgr_sub_image)
        #cv2.imwrite("EdgeDetection/" + image_name_pattern %frame_from, edges_image)

        edges_image2 = Detection_by_tracking._get_part_of_pictures("BackgroudSubtraction/" + image_name_pattern %frame_from, detection[frame_from - 1])
        cv2.imwrite("EdgeDetection_predikcia_vyseky/" + (image_name_pattern % frame_from), edges_image2)

        #edges_image_merge = Detection_by_tracking._merge_image_edge(edges_image2, edges_image)
        #cv2.imwrite("EdgeDetection_spojene_s_predikciou/" + (image_name_pattern % frame_from), edges_image_merge)

        hough_image = _hough_transform_circle(edges_image2, radius_of_object)
        cv2.imwrite("HoughTransform_predikcia_vyseky/" + image_name_pattern % frame_from, hough_image) #°°


        try:
            list_of_coordinates, obr_detegovane = _detect_peaks(hough_image, frame_from, image_name_pattern)
            list_of_frames.append(list_of_coordinates)
            list_of_count_particles.append(len(list_of_coordinates))
        except TypeError:
            print("err")

        _print_progress_bar(i, total, prefix='Progress:', suffix='Complete', length=50)
        frame_from += 1
        i += 1

    _particles_to_XML(list_of_frames, file_name_of_XML)

# ------------------------------------------------------------------------------

def RBC_detection_ellipse(path_source_folder, image_name_pattern, radius_of_object, file_name_of_XML, frame_from,
                          frame_to, stepRotation):
    list_of_frames = []
    list_of_count_particles = []

    background_image = _get_background(path_source_folder, image_name_pattern, frame_from, frame_to)
    i = 1
    total = frame_to - frame_from + 1
    while frame_from <= frame_to:

        count = int(180 / stepRotation)
        j = 0
        while j < count:
            src_image = cv2.imread(path_source_folder + "/" + image_name_pattern % frame_from, 0)
            edges_image = _cany_edge_detection(src_image)
            hough_image = _hough_transform_ellipse(edges_image, radius_of_object, angle)

            list_of_coordinates = _detect_peaks(hough_image, frame_from)
            list_of_frames.append(list_of_coordinates)
            list_of_count_particles.append(len(list_of_coordinates))

            j += 1
            pass

        _print_progress_bar(i, total, prefix='Progress:', suffix='Complete', length=50)
        frame_from += 1
        i += 1

    _particles_to_XML(list_of_frames, file_name_of_XML)

# ------------------------------------------------------------------------------

def _detect_peaks(gray_image, frame_from, image_name_pattern):
    list_of_coordinates = set()
    list_of_correct_points = set()

    mask = cv2.dilate(gray_image, None, iterations=2)
    mask = cv2.threshold(mask, 150, 255, cv2.THRESH_BINARY)[1]
    mask = cv2.erode(mask, None, iterations=1)

    cnts = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = cnts[0] if imutils.is_cv2() else cnts[1]
    if len(cnts) != 0:
        cnts = imutils.contours.sort_contours(cnts)[0]

        width, height = gray_image.shape
        clear_peaks = np.zeros((width, height, 3), np.uint8)

        try:
            for (i, c) in enumerate(cnts):
                ((x, y), radius) = cv2.minEnclosingCircle(c)
                list_of_coordinates.add((int(x), int(y)))

            list_of_correct_points = merge_points(list_of_coordinates, 7)

            for point in list_of_correct_points:
                cv2.circle(clear_peaks, point, 1, (0, 255, 255), 3)
        except TypeError:
            print("type err")

        #cv2.imwrite("DetectObjects/frame_%d.png" % frame_from, clear_peaks)
        #cv2.imwrite("DetectObjects_spojene_s_predikciou/" + image_name_pattern % frame_from + '.png', clear_peaks)
        cv2.imwrite("DetectObjects_predikcia_vyseky/" + image_name_pattern % frame_from, clear_peaks)

        return list_of_correct_points, clear_peaks

# ------------------------------------------------------------------------------

def merge_points(list_of_coordinates, penalty_size):
    penalty_points = set()
    index = 0
    for i in list_of_coordinates:
        index += 1
        accum_x = 0
        accum_y = 0
        penalty_count = 0

        for j in list_of_coordinates:
            if i[0] - penalty_size <= j[0] and j[0] <= i[0] + penalty_size:
                if i[1] - penalty_size <= j[1] and j[1] <= i[1] + penalty_size:
                    accum_x += j[0]
                    accum_y += j[1]
                    penalty_count += 1
                    penalty_points.add(j)

        if penalty_count > 1:
            new_point = (int(accum_x / penalty_count), int(accum_y / penalty_count))
            list_of_coordinates = list_of_coordinates - penalty_points
            list_of_coordinates.add(new_point)

            list_of_coordinates = merge_points(list_of_coordinates, penalty_size)
            return list_of_coordinates

    return list_of_coordinates

# ------------------------------------------------------------------------------

def _hough_transform_circle(gray_image, radius_of_object):
    height, width = gray_image.shape
    parameter_space = np.zeros((height + radius_of_object, width + radius_of_object), np.uint8)
    for x in range(0, height - 1):
        for y in range(0, width - 1):
            if gray_image[x, y] > 127:
                _draw_circle(parameter_space, x, y, radius_of_object, 5)

    #detection_from_tracking(gray_image, parameter_space, radius_of_object, detection)

    parameter_space = parameter_space[0:height, 0:width]  # [y: y + h, x: x + w]
    cv2.normalize(parameter_space, parameter_space, 0, 255, cv2.NORM_MINMAX)

    return parameter_space


# ------------------------------------------------------------------------------

def _hough_transform_ellipse(gray_image, radius_of_object, angle):
    height, width = gray_image.shape
    parameter_space = np.zeros((height + radius_of_object[1], width + radius_of_object[1]), np.uint8)

    radian = np.radians(angle)

    for x in range(0, height):
        for y in range(0, width):
            if gray_image[x, y] > 127:
                drawEllipse(parameter_space, (x, y), radius_of_object, 5, radian)

    parameter_space = parameter_space[0:height, 0:width]  # [y: y + h, x: x + w]
    # cv2.normalize(parameter_space, parameter_space, 0, 255, cv2.NORM_MINMAX)

    return parameter_space


# ------------------------------------------------------------------------------

def _cany_edge_detection(gray_image):
    median = np.median(gray_image)
    edges = cv2.Canny(np.uint8(gray_image), 0.66 * median, 1.33 * median)

    return edges

# ------------------------------------------------------------------------------

def _background_subtraction(gray_image, background_image):
    # placeholder = np.full_like(gray_image, 0, np.int16)
    # placeholder += gray_image
    h = gray_image.shape[0]
    w = gray_image.shape[1]
    gray_image_16 = gray_image.astype(np.int16)
    # for y in range(0, h):
    #     for x in range(0, w):
    #         if gray_image[y, x] <= background_image[y, x]:
    #             gray_image[y, x] = 255 + (gray_image[y, x] - background_image[y, x])
    #         else:
    #             gray_image[y, x] = 0
    background_image_16 = background_image.astype(np.int16)
    gray_image_16 = gray_image_16 - background_image_16
    # # grayFile = open('outputGrayAlgo.txt', "w")
    # np.savetxt(grayFile, gray_image_16)
    gray_image_16 = np.absolute(gray_image_16)
    gray_image = gray_image_16.astype(np.uint8)
    # gray_image = gray_image - background_image

    # for y in range(0, h):
    #     for x in range(0, w):
    #         if gray_image[y, x] < 0:
    #             gray_image[y, x] = 0
    #
    # outputType = np.full_like(gray_image, 0, np.uint8)
    # for y in range(0, h):
    #     for x in range(0, w):
    #         outputType[y, x] = gray_image[y, x]

    # print(outputType.dtype)
    # cv2.threshold(gray_image, 0, 512, cv2.THRESH_TOZERO)
    cv2.normalize(gray_image, gray_image, 0, 255, cv2.NORM_MINMAX)

    return gray_image

def _background_subtraction2(gray_image, background_image):
    cv2.normalize(gray_image, gray_image, 0, 255, cv2.NORM_MINMAX)

    return gray_image


# ------------------------------------------------------------------------------

def _get_background(path_source_folder, image_name_pattern, frame_from, frame_to):
    src_image = cv2.imread(path_source_folder + "/" + image_name_pattern % frame_from, 0)
    background_image = np.float32(src_image)

    frame_from += 1
    n_frames = 2
    while frame_from <= frame_to:
        src_image = cv2.imread(path_source_folder + "/" + image_name_pattern % frame_from, 0)
        frame_from += 1
        n_frames += 1
        cv2.accumulateWeighted(src_image, background_image, 1 / n_frames)

    cv2.imwrite("background.png", background_image)

    return background_image


# ------------------------------------------------------------------------------

def get_backgroundFromVideo(video_path):
    cap = cv2.VideoCapture(video_path)
    background_image = []
    n_frames = 0
    while cap.isOpened():
        ret, frame = cap.read()

        if ret:
            if n_frames == 0:
                background_image = np.float32(frame)
            n_frames += 1
            cv2.accumulateWeighted(frame, background_image, 1 / n_frames)
        else:
            break

    cv2.imwrite("C:\\GitHubCode\\ImageCytometry\\src\\Python\\Detection\\background.png", background_image)

    return background_image


# ------------------------------------------------------------------------------

def _draw_circle(picture, coordinate_X, coordinate_Y, radius, increment):
    a = radius
    b = 0
    err = 0

    while a >= b:
        picture[coordinate_X + a, coordinate_Y + b] += increment
        picture[coordinate_X + b, coordinate_Y + a] += increment
        picture[coordinate_X - b, coordinate_Y + a] += increment
        picture[coordinate_X - a, coordinate_Y + b] += increment
        picture[coordinate_X - a, coordinate_Y - b] += increment
        picture[coordinate_X - b, coordinate_Y - a] += increment
        picture[coordinate_X + b, coordinate_Y - a] += increment
        picture[coordinate_X + a, coordinate_Y - b] += increment

        if err <= 0:
            b += 1;
            err += 2 * b + 1
        else:
            a -= 1
            err -= 2 * a + 1
        pass


# ------------------------------------------------------------------------------

def drawEllipse(picture, origins, radius, increment, angle):
    w_2 = radius[0] * radius[0]
    h_2 = radius[1] * radius[1]
    fw2 = 4 * w_2
    fh2 = 4 * h_2

    # first half
    x = 0
    y = radius[1]
    sigma = 2 * h_2 + w_2 * (1 - 2 * radius[1])
    while h_2 * x <= w_2 * y:
        picture[rotation((origins[0] + x, origins[1] + y), origins, angle)] += increment
        picture[rotation((origins[0] - x, origins[1] + y), origins, angle)] += increment
        picture[rotation((origins[0] + x, origins[1] - y), origins, angle)] += increment
        picture[rotation((origins[0] - x, origins[1] - y), origins, angle)] += increment

        if sigma >= 0:
            sigma += fw2 * (1 - y)
            y -= 1

        sigma += h_2 * ((4 * x) + 6)

        x += 1
        pass

    # second half
    x = radius[0]
    y = 0
    sigma = 2 * w_2 + h_2 * (1 - 2 * radius[0])
    while w_2 * y <= h_2 * x:
        picture[rotation((origins[0] + x, origins[1] + y), origins, angle)] += increment
        picture[rotation((origins[0] - x, origins[1] + y), origins, angle)] += increment
        picture[rotation((origins[0] + x, origins[1] - y), origins, angle)] += increment
        picture[rotation((origins[0] - x, origins[1] - y), origins, angle)] += increment

        if sigma >= 0:
            sigma += fh2 * (1 - x)
            x -= 1

        sigma += w_2 * ((4 * y) + 6)

        y += 1
        pass


# ------------------------------------------------------------------------------

def _particles_to_XML(list_of_frames, file_name_of_XML):
    XML_root = ET.Element("RBC_detection", framesCount=str(len(list_of_frames)))

    for x in range(len(list_of_frames)):
        frame = ET.SubElement(XML_root, "frame", number=str(x), particlesCount=str(len(list_of_frames[x])))
        index = 0
        for y in list_of_frames[x]:
            ET.SubElement(frame, "particle", number=str(index), x=str(y[0]), y=str(y[1]))
            index += 1

    tree = ET.ElementTree(XML_root)
    tree.write(file_name_of_XML)

# ------------------------------------------------------------------------------

def _print_progress_bar(iteration, total, prefix="", suffix="", decimals=1, length=100, fill="█"):
    percent = ("{0:." + str(decimals) + "f}").format(100 * (iteration / float(total)))
    filled_length = int(length * iteration // total)
    bar = fill * filled_length + '-' * (length - filled_length)

    print('\r%s |%s| %s%% %s' % (prefix, bar, percent, suffix), end='\r')

    if iteration == total:
        print()


# ------------------------------------------------------------------------------

def _video_parser(path_source_folder, path_destination_folder, image_name_pattern, time_from, time_to):
    cap = cv2.VideoCapture(path_source_folder)
    fps = cap.get(5)  # cv2.cv.CV_CAP_PROP_FPS = 5

    frame_from = time_from * fps
    frame_to = frame_from + ((time_to - time_from) * fps)

    i = 1
    while frame_from <= frame_to:
        ret, frame = cap.read()
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        cv2.imwrite(path_destination_folder + "/" + image_name_pattern % i, frame)
        frame_from += 1
        i += 1


# ------------------------------------------------------------------------------

def renameFiles(path_source_folder, path_destination_folder, image_name_pattern, frame_from, frame_to):
    i = 1
    while frame_from <= frame_to:
        src_image = cv2.imread(path_source_folder + "/" + image_name_pattern % frame_from)
        if src_image != None:
            cv2.imwrite(path_destination_folder + "/" + image_name_pattern % i, src_image)
            i += 1

        frame_from += 1
        pass


# ------------------------------------------------------------------------------

#run("SourceImages", "frame_%d.jpg", 12, "export.xml", 1, 120)
# "SourceImages" - (path) source folder
# "frame_%d.jpg" - (image) name pattern
# 12 - radius of circle
# "export.xml" - file name of XML (export)
# 1 - frame from
# 120 - frame to