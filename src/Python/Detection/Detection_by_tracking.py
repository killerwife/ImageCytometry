import numpy as np
import cv2

#import RBC_detection_full_script
#	------------------------------------------------------------------------------

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
            print('Number: ',f.attrib['number'] ,' len: ', len(detection), ' add: ', count_to_add)
            for i in range(count_to_add):
                pom = []
                detection.append(pom)
            print('dlzka: ', len(detection))
        frame = []
        for cell in f:
            x = cell.attrib['x']
            y = cell.attrib['y']
            frame.append([x,y])
        detection.append(frame)
    return detection

#	------------------------------------------------------------------------------

def _hough_transform_circle(gray_image, radius_of_object, detection, image_name_pattern, frame):
    height, width = gray_image.shape
    parameter_space = np.zeros((height + radius_of_object, width + radius_of_object), np.uint8)
    for cell in range(len(detection)): # TODO vstupom bude left_top alebo stred?
        x_left = int(detection[cell][1]) # x a y sa berie naopak
        y_left = int(detection[cell][0])
        for x in range(x_left, x_left + 30): # 25 ? # TODO prerobit obe na stred
            for y in range(y_left, y_left + 30): # 25 ?
                if gray_image[x,y] > 127:
                    _draw_circle(parameter_space, x, y, radius_of_object, 10)

    parameter_space = parameter_space[0:height, 0:width]  # [y: y + h, x: x + w]
    cv2.normalize(parameter_space, parameter_space, 0, 255, cv2.NORM_MINMAX)

    cv2.imwrite("HoughTransform2/" + (image_name_pattern % frame) + '.png', parameter_space)

    return parameter_space

#	------------------------------------------------------------------------------
def _draw_rectangle(source_picture, detection_cells):
    #picture = cv2.imread(source_picture)
    picture = source_picture

    x, y = picture.shape
    img_tracks = np.zeros((int(x), int(y)), np.uint8)  # vytvori cierny obrazok velkost: x*y

    for img in range(len(detection_cells)):
        x = int(detection_cells[img][0])  # TODO tu bola chyba - bolo to naopak
        y = int(detection_cells[img][1])
        #picture1 = picture[x - 10:x + 40, y - 10:y + 40]
        #picture1 = _cany_edge_detection2(picture1, "cell1", 1)
        #img_tracks[x - 10:x + 40, y - 10:y + 40] = picture1
        cv2.rectangle(img_tracks, (x-15,y-15), (x+15,y+15),(255,255,255),2 )

        #cv2.imshow("draw", img_tracks)
        #cv2.waitKey(0)

    return img_tracks

#	------------------------------------------------------------------------------
def _get_part_of_pictures(source_picture, detection_cells):
    picture = cv2.imread(source_picture)

    x, y, c = picture.shape
    img_tracks = np.zeros((int(x), int(y)), np.uint8) # vytvori cierny obrazok velkost: x*y

    for img in range(len(detection_cells)):
        x = int (detection_cells[img][1]) # TODO tu bola chyba - bolo to naopak
        y = int(detection_cells[img][0])
        picture1 = picture[x - 15:x + 15, y - 15:y + 15]
        picture1 = _cany_edge_detection(picture1, "cell1", 1)
        img_tracks[x - 15:x + 15, y - 15:y + 15] = picture1

    return img_tracks


#	------------------------------------------------------------------------------

def _cany_edge_detection(gray_image, image_name_pattern, frame):
    median = np.median(gray_image)
    edges = cv2.Canny(np.uint8(gray_image), 0.3 * median, 1.2 * median)
    #edges = cv2.Canny(np.uint8(gray_image), 0.2 * median, 1.0 * median)
    #cv2.imwrite("EdgeDetection_predikcia/" + (image_name_pattern % frame) + ".png", edges)
    #cv2.imwrite("EdgeDetection2/" + (image_name_pattern % frame) + ".png", edges)

    return edges


#	------------------------------------------------------------------------------

def _merge_image_edge(image1, image2): # spajanie dvoch obrazkov (prekrytie) -png formaty
    img1 = image1
    img2 = image2

    result = cv2.addWeighted(img1, 1, img2, 1, 0)  # TODO nastavit parametre
    return result


#	------------------------------------------------------------------------------

def _merge_image2(image1, image2): # spajanie dvoch obrazkov (prekrytie) -png formaty
    img1 = image1
    img2 = image2

    rows, cols  = img2.shape
    roi = img1[0:rows, 0:cols]

    #-ret, mask = cv2.threshold(img2, 58, 255, cv2.THRESH_BINARY)
    #-img2_fg = cv2.bitwise_and(img2, img2, mask=mask)

    #-img2_fg = cv2.erode(mask, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)), iterations=1)

    #-dst = cv2.add(img1, img2_fg)
    #-img1[0:rows, 0:cols] = dst

    #ret, mask = cv2.threshold(img2, 10, 255, cv2.THRESH_BINARY)
    #img2_fg = cv2.bitwise_and(img2, img2, mask=mask)

    #print('1 >> ', img1.shape, ' 2 >> ', img2.shape  )
    result = cv2.addWeighted(img2, 1, img1, 1, 0) #TODO nastavit parametre
    #@result = np.uint8(cv2.addWeighted(img2, 255.0, img1, 255.0, 0.0))
    #@@result = np.concatenate((img1, img1), axis=1)
    #@cv2.imwrite('result.png', result)
    return result


#	------------------------------------------------------------------------------


#	------------------------------------------------------------------------------

def _merge_images(image1, image2): # spajanie dvoch obrazkov (prekrytie) -png formaty
    #img1 = cv2.imread(image1)
    #img2 = cv2.imread(image2)

    img1 = image1
    img2 = image2

    rows, cols  = img2.shape
    roi = img1[0:rows, 0:cols]

    #img2gray = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
    #ret, mask = cv2.threshold(img2, 10, 255, cv2.THRESH_BINARY)
    #mask_inv = cv2.bitwise_not(mask)
    #img1_bg = cv2.bitwise_and(roi, roi, mask=mask_inv)
    #img2_fg = cv2.bitwise_and(img2, img2, mask=mask)
    #dst = cv2.add(img1_bg, img2_fg)
    #img1[0:rows, 0:cols] = dst

    thresh = 0
    maxValue = 255
    th, dst = cv2.threshold(img2, thresh, maxValue, cv2.THRESH_BINARY)


    ret, mask = cv2.threshold(img2, 58, 255, cv2.THRESH_BINARY)
    img2_fg = cv2.bitwise_and(img2, img2, mask=mask)

    img2_fg = cv2.erode(mask, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)), iterations=1)

    dst = cv2.add(img1, img2_fg)
    img1[0:rows, 0:cols] = dst


    #@cv2.imshow("1", img1)
    #@cv2.imshow("2", img2)
    #@cv2.waitKey(0)

    #ret, mask = cv2.threshold(img2, 10, 255, cv2.THRESH_BINARY)
    #img2_fg = cv2.bitwise_and(img2, img2, mask=mask)

    #print('1 >> ', img1.shape, ' 2 >> ', img2.shape  )
    #result = cv2.addWeighted(img2_fg, 1, img1, 1, 0) #TODO nastavit parametre
    #@result = np.uint8(cv2.addWeighted(img2, 255.0, img1, 255.0, 0.0))
    #@@result = np.concatenate((img1, img1), axis=1)
    #@cv2.imwrite('result.png', result)
    return img1


#	------------------------------------------------------------------------------

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

