import cv2
import numpy as np
from time import sleep
import xml.etree.cElementTree as ET
import imutils
from imutils import contours
import os

# *********************************************************************************************

def run(file_name):
    detection = []
    detection = _parse_from_XML(file_name)

    #draw_circles(detection)
    _merge_image()
    #_draw_rectangle(detection)

# *********************************************************************************************

def draw_circles(detection):
    for frame in range(len(detection)):
        clear_peaks = np.zeros((720, 1280, 3), np.uint8)

        for point in range(len(detection[frame])):
            x = int(detection[frame][point][0])
            y = int(detection[frame][point][1])
            #print(detection[frame][point][0] + " - " + detection[frame][point][1])
            cv2.circle(clear_peaks, (x,y), 1, (0, 255, 0), 3) #(img, center, radius, color, thickness)
        cv2.imwrite("zaradene_bunky/" + 'frame' + str(frame) + '.png', clear_peaks)

# *********************************************************************************************

def _draw_rectangle(detection):
    frame_from = 1
    frame_to = 120
    while frame_from <= frame_to:
        picture = cv2.imread("spojene_bunky/frame" + str(frame_from) + ".png")
        #picture = source_picture

        x, y, alfa = picture.shape
        img_tracks = np.zeros((int(x), int(y)), np.uint8)  # vytvori cierny obrazok velkost: x*y

        for img in range(len(detection[frame_from])):
            x = int(detection[frame_from][img][1])  # TODO tu bola chyba - bolo to naopak
            y = int(detection[frame_from][img][0])
            cv2.rectangle(img_tracks, (x - 15, y - 15), (x + 15, y + 15), (255, 255, 255), 2)

        cv2.imwrite("predikcia/" + 'frame' + str(frame_from + 1) + '.png', img_tracks)
        frame_from += 1

    #return img_tracks

# *********************************************************************************************

def _parse_from_XML(file_name_of_XML): # matica - zacina od 0 !!!
    import xml.etree.ElementTree as ET
    tree = ET.parse(file_name_of_XML)
    root = tree.getroot()

    detection = []
    detection.append([]) # TODO nulty.. treba?
    for f in root: # TODO riesit aj cislo framu !!!
        count_to_add = 0
        if(int(f.attrib['number']) > len(detection)):
            count_to_add = int(f.attrib['number']) - len(detection) #- 1
            for i in range(count_to_add):
                pom = []
                detection.append(pom)
        frame = []
        for cell in f:
            x = cell.attrib['x']
            y = cell.attrib['y']
            frame.append([x,y])
        detection.append(frame)
    return detection


# *********************************************************************************************

def _merge_image(): # spajanie dvoch obrazkov (prekrytie) -png formaty
    #img1 = image1
    #img2 = image2

    frame_from = 1
    frame_to = 120
    while frame_from <= frame_to:

        img1 = cv2.imread("VYSTUP_Z_TRASOVANIA/spojene_bunky/frame" + str(frame_from) + ".png")
        img2 = cv2.imread("s_pozadim_len_predikcia/frame" + str(frame_from) + ".png")

        if(os.path.exists("s_pozadim_len_predikcia/frame" + str(frame_from) + ".png")):
            try:
                #rows, cols  = img2.shape
                #rows, cols, alfa = img2.shape
                #roi = img1[0:rows, 0:cols]

                result = cv2.addWeighted(img2, 0.7, img1, 0.3, 0)
                cv2.imwrite("spojene_len_predikcia_s_povodnym/frame" + str(frame_from) + ".png", result)
            except TypeError:
                print("err")

        frame_from += 1

# *********************************************************************************************
# *********************************************************************************************

#run("zaradene_v_trasach2.xml") # nazov suboru na vstup
# run("nasledovne_bunky_tras.xml") # nazov suboru na vstup
run("predikcia_06_04.xml")
