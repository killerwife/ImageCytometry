import cv2
import math
import random
import numpy as np

def draw_tracks(tracks, img, first=-1, last=-1):
    print('draw tracks: drawing...')
    if ((first == -1) and (last == -1)):
        # chceme vykreslit vsetky tracky
        print('Tracks: all: ' + str(len(tracks)))
        for i in range(len(tracks)):
            color = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
            for j in range(len(tracks[i]) - 1):
                cv2.line(img, (int(tracks[i][j][0])*3, int(tracks[i][j][1])*3), ((int(tracks[i][j + 1][0])*3, int(tracks[i][j + 1][1])*3)), color,
                        1)
                # cv2.putText(img,s,(x+5,y+5),cv2.FONT_HERSHEY_PLAIN,0.9,(255,255,255))

    elif ((first != -1) and (last == -1)):
        # chceme vykreslit jeden konkretny track
        print('Track: ' + str(first))
        color = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
        # print(len(tracks[first]))
        for j in range(len(tracks[first]) - 1):
            cv2.line(img, (int(tracks[first][j][0]), int(tracks[first][j][1])),
                     ((int(tracks[first][j + 1][0]), int(tracks[first][j + 1][1]))),(255,255,255), 1)
    else:
        # chceme vykreslit seq trackov od - do
        print('Tracks: ' + str(first) + '-' + str(last))
        for i in range(first, (last + 1)):
            color = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
            for j in range(len(tracks[i]) - 1):
                cv2.line(img, (int(tracks[i][j][0]), int(tracks[i][j][1])), (int(tracks[i][j + 1][0]), int(tracks[i][j + 1][1])), color,
                         1)

    return img
# ---------------------------------------------------------------------------------------------
def get_distance(a, b):
    return math.sqrt((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2)
# ----------------------------------------------------------------------
def get_velocity(point_a, point_b):
    distance = get_distance(point_a, point_b)
    t = math.fabs(point_b[2]-point_a[2])
    v = distance / t
    return v
# --------------------------------------------------------------------------------------------
def parse_xml_tracks(file_name):
    xml_tracks = []
    import xml.etree.cElementTree as element
    tree = element.parse(file_name)

    for elt in tree.getiterator('track'):
        # print('tracks '+str(len(xml_tracks)))

        track = []
        for par in elt.getiterator('particle'):
            # particleCount += 1
            x = (float(par.get('x')))
            y = (float(par.get('y')))
            t = float(par.get('time'))
            track.append([x,y,t])
        # print(len(track))
        xml_tracks.append(track)

    return xml_tracks
# --------------------------------------------------------------------------
def _get_intersections(hnc, count):
    hncs = []
    for i in range(hnc-count, hnc,+1):
        # print(i)
        hncs.append(int(i))

    for i in range(hnc,count+hnc+1,+1):
        # print(i)
        hncs.append(int(i))


    return hncs
# ---------------------------------------------------------------------------
def get_results_simulation_check(xml_tracks):

    la1 = 27
    lb1 = 28
    lc1 = -2705
    c1 = 0

    la2 = 0
    lb2 = 1
    lc2 = -100
    c2 = 0

    la3 = 1
    lb3 = 1
    lc3 = -262
    c3 = 0

    data = []
    for i in range(len(xml_tracks)):

        for j in range(len(xml_tracks[i])-1):
            # kontrola prvej hranice
            x1 = xml_tracks[i][j][0]
            y1 = xml_tracks[i][j][1]
            x2 = xml_tracks[i][j+1][0]
            y2 =xml_tracks[i][j+1][1]
            if( _up_line(la1,lb1,lc1,x1,y1) == True and _up_line(la1,lb1,lc1,x2,y2) == False):
                print('bod presiel 1. hranicu '+str(i))
                print(xml_tracks[i][j])
                print(xml_tracks[i][j+1])

                # img_sim_tracks = draw_tracks(xml_tracks, np.zeros((200, 160, 3), np.uint8), i, -1)
                # cv2.line(img_sim_tracks, (95, 5), (67, 32), (0, 255, 255), 1)
                # cv2.imshow('tracks ' + str(i)+' h1', img_sim_tracks)

                velocity = get_velocity(xml_tracks[i][j],xml_tracks[i][j+1])
                cor_x = ((xml_tracks[i][j][0]+xml_tracks[i][j+1][0])/2)
                cor_y = ((xml_tracks[i][j][1]+xml_tracks[i][j+1][1])/2)
                data.append([cor_x,cor_y, velocity])
                c1 +=1
            if (_up_line(la2,lb2,lc2, x1, y1) == True and _up_line(la2,lb2,lc2,x2,y2) == False):
                print('bod tracku '+str(i)+' presiel 2. hranicu '+str(i))
                print(xml_tracks[i][j])
                print(xml_tracks[i][j + 1])

                # img_sim_tracks = draw_tracks(xml_tracks, np.zeros((200, 160, 3), np.uint8),i,-1)
                # cv2.line(img_sim_tracks, (67, 100), (95, 100), (0, 255, 255), 1)
                # cv2.imshow('tracks ' + str(i) + ' h2', img_sim_tracks)

                velocity = get_velocity(xml_tracks[i][j], xml_tracks[i][j + 1])
                cor_x = ((xml_tracks[i][j][0] + xml_tracks[i][j + 1][0]) / 2)
                cor_y = ((xml_tracks[i][j][1] + xml_tracks[i][j + 1][1]) / 2)
                data.append([cor_x, cor_y, velocity])
                c2 += 1
            if (_up_line(la3,lb3,lc3, x1, y1) == True and _up_line(la3,lb3,lc3,x2,y2) == False):
                print('bod presiel 3. hranicu '+str(i))
                print(xml_tracks[i][j])
                print(xml_tracks[i][j + 1])

                # img_sim_tracks = draw_tracks(xml_tracks, np.zeros((200, 160, 3), np.uint8), i, -1)
                # cv2.line(img_sim_tracks, (67, 195), (95, 167), (0, 255, 255), 1)
                # cv2.imshow('tracks ' + str(i) + ' h3', img_sim_tracks)

                velocity = get_velocity(xml_tracks[i][j], xml_tracks[i][j + 1])
                cor_x = ((xml_tracks[i][j][0] + xml_tracks[i][j + 1][0]) / 2)
                cor_y = ((xml_tracks[i][j][1] + xml_tracks[i][j + 1][1]) / 2)
                data.append([cor_x, cor_y, velocity])
                c3 += 1
    print('H1 = ' + str(c1))
    print('H2 = ' + str(c2))
    print('H3 = ' + str(c3))

    # print(data)
    return data
# ------------------------------------------------------------------------------
def get_results_simulation(xml_tracks,count):
    h1a = 27
    h1b = 28
    h1c = -2705

    h2a = 0
    h2b = 1
    h2c = -100

    h3a = 1
    h3b = 1
    h3c = -262

    count = count
    h1cs = _get_intersections(h1c, count)
    h2cs = _get_intersections(h2c, count)
    h3cs = _get_intersections(h3c, count)

    inf_h1 = []
    inf_h2 = []
    inf_h3 = []

    dataH1 = []
    dataH2 = []
    dataH3 = []

    for i in range(len(xml_tracks)):
        temp_trac_velocity1 = []
        temp_trac_velocity2 = []
        temp_trac_velocity3 = []
        for j in range(len(xml_tracks[i])-1):
            x1 = xml_tracks[i][j][0]
            y1 = xml_tracks[i][j][1]
            x2 = xml_tracks[i][j + 1][0]
            y2 = xml_tracks[i][j + 1][1]

            # kontrola prvej hranice -------------------------------------------------------------------
            # cez vsetky priesecniky oblasti
            for p in range(len(h1cs)):
                if(_up_line(h1a,h1b,h1cs[p],x1,y1) == True and _up_line(h1a,h1b,h1cs[p],x2,y2) == False):
                    # print('bod presiel 1 hranicu' + str(i))
                    # print(xml_tracks[i][j])
                    # print(xml_tracks[i][j + 1])
                    velocity = get_velocity(xml_tracks[i][j], xml_tracks[i][j + 1])
                    temp_trac_velocity1.append(velocity)
                    if (inf_h1.count(i) == 0):
                        inf_h1.append(i)


            #KONTROLA DRUHEJ HRANICE------------------------------------------------------------------------------
            for p in range(len(h2cs)):
                if(_up_line(h2a,h2b,h2cs[p],x1,y1) == True and _up_line(h2a,h2b,h2cs[p],x2,y2) == False):
                    # print('bod presiel 2 hranicu' + str(i))
                    # print(xml_tracks[i][j])
                    # print(xml_tracks[i][j + 1])
                    velocity = get_velocity(xml_tracks[i][j], xml_tracks[i][j + 1])
                    temp_trac_velocity2.append(velocity)
                    if (inf_h2.count(i) == 0):
                        inf_h2.append(i)

            # KONTROLA TRETEJ HRANICE ------------------------------------------------------------------------------------
            for p in range(len(h3cs)):
                if(_up_line(h3a,h3b,h3cs[p],x1,y1) == True and _up_line(h3a,h3b,h3cs[p],x2,y2) == False):
                    # print('bod presiel 3 hranicu' + str(i))
                    # print(xml_tracks[i][j])
                    # print(xml_tracks[i][j + 1])
                    velocity = get_velocity(xml_tracks[i][j], xml_tracks[i][j + 1])
                    temp_trac_velocity3.append(velocity)
                    if (inf_h3.count(i) == 0):
                        inf_h3.append(i)

        if(len(temp_trac_velocity1) > 0):
            avg_velocity = sum(temp_trac_velocity1)/len(temp_trac_velocity1)
            dataH1.append(avg_velocity)
            # print('avg za oblast 1 = '+ str(avg_velocity))

        if (len(temp_trac_velocity2) > 0):
            avg_velocity = sum(temp_trac_velocity2) / len(temp_trac_velocity2)
            dataH2.append(avg_velocity)
            # print('avg za oblast 2 = ' + str(avg_velocity))

        if (len(temp_trac_velocity3) > 0):
            avg_velocity = sum(temp_trac_velocity3) / len(temp_trac_velocity3)
            dataH3.append(avg_velocity)
            # print('avg za oblast 3 = ' + str(avg_velocity))


    # print(inf_h1)
    # print(inf_h2)
    # print(inf_h3)
    return dataH1,dataH2,dataH3

# ------------------------------------------------------------------------------
def get_results_video_check(xml_tracks):
    # zmenene hodnoty podla ph
    # kontrolovat tracky iba v polovici
    global pixel_size
    la1 = 321
    lb1 = -324
    lc1 = -75908
    c1 = 0

    la11 = 318
    lb11 = -321
    lc11 = -75304
    c11 = 0

    la12 = 324
    lb12 = -327
    lc12 = -76510
    c12 = 0

    la2 = 0
    lb2 = +3
    lc2 = -350
    c2 = 0

    la21 = 0
    lb21 = +3
    lc21 = -349
    c21 = 0

    la22 = 0
    lb22 = +1
    lc22 = -117
    c22 = 0


    la3 = 1
    lb3 = -1
    lc3 = -64
    c3 = 0

    la31 = 3
    lb31 = -3
    lc31 = -193
    c31 = 0

    la32 = 3
    lb32 = -3
    lc32 = -191
    c32 = 0


    h1=[]
    h2=[]
    h3=[]

    data = []

    for i in range(len(xml_tracks)):

        for j in range(len(xml_tracks[i])-1):
            x1 = xml_tracks[i][j][0]
            y1 = xml_tracks[i][j][1]
            x2 = xml_tracks[i][j + 1][0]
            y2 = xml_tracks[i][j + 1][1]

            # kontrola prvej hranice -------------------------------------------------------------------
            if (_up_line(la11, lb11,lc11, x1, y1) == True and _up_line(la11,lb11,lc11,x2,y2) == False and h1.count(i)==0):
                    # print('bod presiel 1.1. hranicu'+str(i))
                    # print(xml_tracks[i][j])
                    # print(xml_tracks[i][j + 1])
                    # img = draw_tracks(xml_tracks, np.zeros((int(resolution_y * pixel_size), int(resolution_x * pixel_size), 3), np.uint8), i, -1)
                    # cv2.line(img, (int(773 * pixel_size), int(62 * pixel_size)),  (int(880 * pixel_size), int(168 * pixel_size)), (0, 255, 255), 1)
                    # cv2.imshow('tracks ' + str(i) + ' H1.1', img)

                    velocity = get_velocity(xml_tracks[i][j], xml_tracks[i][j + 1])
                    cor_x = ((xml_tracks[i][j][0] + xml_tracks[i][j + 1][0]) / 2)
                    cor_y = ((xml_tracks[i][j][1] + xml_tracks[i][j + 1][1]) / 2)
                    data.append([cor_x, cor_y, velocity])
                    c11 += 1
                    # print(str(i) + ' h11 ---- ' + str(x1) + ' ' + str(y1) + ' ++++++ ' + str(x2) + ' ' + str(y2))
                    h1.append(i)


            elif( _up_line(la1,lb1,lc1,x1,y1) == True and _up_line(la1,lb1,lc1,x2,y2) == False and h1.count(i)==0):
                # print('bod presiel 1. hranicu'+str(i))
                # print(xml_tracks[i][j])
                # print(xml_tracks[i][j+1])

                # img = draw_tracks(xml_tracks, np.zeros((int(resolution_y * pixel_size), int(resolution_x * pixel_size), 3), np.uint8), i, -1)
                # cv2.line(img, (int(772 * pixel_size), int(62 * pixel_size)), (int(880 * pixel_size), int(169 * pixel_size)), (0, 255, 255), 1)
                # cv2.imshow('tracks ' + str(i)+' H1', img)

                velocity = get_velocity(xml_tracks[i][j],xml_tracks[i][j+1])
                cor_x = ((xml_tracks[i][j][0]+xml_tracks[i][j+1][0])/2)
                cor_y = ((xml_tracks[i][j][1]+xml_tracks[i][j+1][1])/2)
                data.append([cor_x,cor_y, velocity])
                c1 +=1
                h1.append(i)
                # print(str(i)+' h1---- '+ str(x1)+' '+str(y1)+' ++++++ '+ str(x2)+' '+str(y2))

            elif (_up_line(la12, lb12,lc12, x1, y1) == True and _up_line(la12,lb12,lc12,x2,y2) == False and h1.count(i)==0):
                    # print('bod presiel 1.2. hranicu'+str(i))
                    # print(xml_tracks[i][j])
                    # print(xml_tracks[i][j + 1])
                    # img = draw_tracks(xml_tracks, np.zeros((int(resolution_y * pixel_size), int(resolution_x * pixel_size), 3), np.uint8), i, -1)
                    # cv2.line(img, (int(771 * pixel_size), int(62 * pixel_size)),  (int(880 * pixel_size), int(170 * pixel_size)), (0, 255, 255), 1)
                    # cv2.imshow('tracks ' + str(i) + ' H1.2', img)

                    velocity = get_velocity(xml_tracks[i][j], xml_tracks[i][j + 1])
                    cor_x = ((xml_tracks[i][j][0] + xml_tracks[i][j + 1][0]) / 2)
                    cor_y = ((xml_tracks[i][j][1] + xml_tracks[i][j + 1][1]) / 2)
                    data.append([cor_x, cor_y, velocity])
                    c12 += 1
                    # print(str(i) + ' h12 ----- ' + str(x1) + ' ' + str(y1) + ' ++++++ ' + str(x2) + ' ' + str(y2))
                    h1.append(i)


            #         KONTROLA DRUHEJ HRANICE------------------------------------------------------------------------------

            elif (_up_line(la21, lb21,lc21, x1, y1) == False and _up_line(la21,lb21,lc21,x2,y2) == True and h2.count(i)==0):
                if (x1 > (600 * pixel_size)):
                    # print('bod presiel 2.1 hranicu'+str(i))
                    # print(xml_tracks[i][j])
                    # print(xml_tracks[i][j + 1])
                    # img = draw_tracks(xml_tracks, np.zeros((int(resolution_y * pixel_size), int(resolution_x * pixel_size), 3), np.uint8), i, -1)
                    # cv2.line(img, (int(772 * pixel_size), int(349 * pixel_size)),  (int(880 * pixel_size), int(349 * pixel_size)), (0, 255, 255), 1)
                    # cv2.imshow('tracks ' + str(i) + ' H2.1', img)

                    velocity = get_velocity(xml_tracks[i][j], xml_tracks[i][j + 1])
                    cor_x = ((xml_tracks[i][j][0] + xml_tracks[i][j + 1][0]) / 2)
                    cor_y = ((xml_tracks[i][j][1] + xml_tracks[i][j + 1][1]) / 2)
                    data.append([cor_x, cor_y, velocity])
                    c21 += 1
                    # print(str(i) + ' h21 ----- ' + str(x1) + ' ' + str(y1) + ' ++++++ ' + str(x2) + ' ' + str(y2))
                    h2.append(i)

            elif (_up_line(la2, lb2,lc2, x1, y1) == False and _up_line(la2,lb2,lc2,x2,y2) == True and h2.count(i)==0 ):
                if (x1 > (600 * pixel_size)):
                    # print('bod presiel 2. hranicu'+str(i))
                    # print(xml_tracks[i][j])
                    # print(xml_tracks[i][j + 1])

                    # img = draw_tracks(xml_tracks, np.zeros((int(resolution_y*pixel_size), int(resolution_x*pixel_size), 3), np.uint8),i,-1)
                    # cv2.line(img, (int(772 * pixel_size), int(350 * pixel_size)),(int(880 * pixel_size), int(350 * pixel_size)), (0, 255, 255), 1)
                    # cv2.imshow('tracks '+str(i)+' H2', img)

                    velocity = get_velocity(xml_tracks[i][j], xml_tracks[i][j + 1])
                    cor_x = ((xml_tracks[i][j][0] + xml_tracks[i][j + 1][0]) / 2)
                    cor_y = ((xml_tracks[i][j][1] + xml_tracks[i][j + 1][1]) / 2)
                    data.append([cor_x, cor_y, velocity])
                    c2 += 1
                    # print(str(i) + ' h2 ----- ' + str(x1) + ' ' + str(y1) + ' ++++++ ' + str(x2) + ' ' + str(y2))
                    h2.append(i)

            elif (_up_line(la22, lb22,lc22, x1, y1) == False and _up_line(la22,lb22,lc22,x2,y2) == True and h2.count(i)==0):
                if (x1 > (600 * pixel_size)):
                    # print('bod presiel 2.2 hranicu'+str(i))
                    # print(xml_tracks[i][j])
                    # print(xml_tracks[i][j + 1])
                    # img = draw_tracks(xml_tracks, np.zeros((int(resolution_y * pixel_size), int(resolution_x * pixel_size), 3), np.uint8), i, -1)
                    # cv2.line(img, (int(772 * pixel_size), int(351 * pixel_size)),  (int(880 * pixel_size), int(351 * pixel_size)), (0, 255, 255), 1)
                    # cv2.imshow('tracks ' + str(i) + ' H2.2', img)

                    velocity = get_velocity(xml_tracks[i][j], xml_tracks[i][j + 1])
                    cor_x = ((xml_tracks[i][j][0] + xml_tracks[i][j + 1][0]) / 2)
                    cor_y = ((xml_tracks[i][j][1] + xml_tracks[i][j + 1][1]) / 2)
                    data.append([cor_x, cor_y, velocity])
                    c22 += 1
                    # print(str(i) + ' h22 ----- ' + str(x1) + ' ' + str(y1) + ' ++++++ ' + str(x2) + ' ' + str(y2))
                    h2.append(i)

            # KONTROLA TRETEJ HRANICE ------------------------------------------------------------------------------------

            elif (_up_line(la31, lb31,lc31, x1, y1) == True and _up_line(la31,lb31,lc31,x2,y2) == False and h3.count(i)==0):
                if (x1 > (600 * pixel_size)):
                    # print('bod presiel 3.1 hranicu'+str(i))
                    # print(xml_tracks[i][j])
                    # print(xml_tracks[i][j + 1])
                    # img = draw_tracks(xml_tracks, np.zeros((int(resolution_y * pixel_size), int(resolution_x * pixel_size), 3), np.uint8), i, -1)
                    # cv2.line(img, (int(771 * pixel_size), int(62 * pixel_size)),  (int(880 * pixel_size), int(170 * pixel_size)), (0, 255, 255), 1)
                    # cv2.imshow('tracks ' + str(i) + ' H3.1', img)

                    velocity = get_velocity(xml_tracks[i][j], xml_tracks[i][j + 1])
                    cor_x = ((xml_tracks[i][j][0] + xml_tracks[i][j + 1][0]) / 2)
                    cor_y = ((xml_tracks[i][j][1] + xml_tracks[i][j + 1][1]) / 2)
                    data.append([cor_x, cor_y, velocity])
                    c31 += 1
                    # print(str(i) + ' h31 ----- ' + str(x1) + ' ' + str(y1) + ' ++++++ ' + str(x2) + ' ' + str(y2))
                    h3.append(i)

            elif (_up_line(la3, lb3,lc3, x1, y1) == True and _up_line(la3,lb3,lc3,x2,y2) == False and h3.count(i)==0):
               if(x1 > (600*pixel_size)):
                    # print('bod presiel 3. hranicu'+str(i))
                    # print(xml_tracks[i][j])
                    # print(xml_tracks[i][j + 1])
                    # img = draw_tracks(xml_tracks, np.zeros((int(resolution_y * pixel_size), int(resolution_x * pixel_size), 3), np.uint8), i, -1)
                    # cv2.line(img, (int(772 * pixel_size), int(580 * pixel_size)),  (int(880 * pixel_size), int(688 * pixel_size)), (0, 255, 255), 1)
                    # cv2.imshow('tracks ' + str(i) + ' H3', img)

                    velocity = get_velocity(xml_tracks[i][j], xml_tracks[i][j + 1])
                    cor_x = ((xml_tracks[i][j][0] + xml_tracks[i][j + 1][0]) / 2)
                    cor_y = ((xml_tracks[i][j][1] + xml_tracks[i][j + 1][1]) / 2)
                    data.append([cor_x, cor_y, velocity])
                    c3 += 1
                    # print(str(i) + ' h3 ----- ' + str(x1) + ' ' + str(y1) + ' ++++++ ' + str(x2) + ' ' + str(y2))
                    h3.append(i)


            elif (_up_line(la32, lb32,lc32, x1, y1) == True and _up_line(la32,lb32,lc32,x2,y2) == False and h3.count(i)==0):
                if (x1 > (600 * pixel_size)):
                    # print('bod presiel 3.2 hranicu'+str(i))
                    # print(xml_tracks[i][j])
                    # print(xml_tracks[i][j + 1])
                    # img = draw_tracks(xml_tracks, np.zeros((int(resolution_y * pixel_size), int(resolution_x * pixel_size), 3), np.uint8), i, -1)
                    # cv2.line(img, (int(771 * pixel_size), int(62 * pixel_size)),  (int(880 * pixel_size), int(170 * pixel_size)), (0, 255, 255), 1)
                    # cv2.imshow('tracks ' + str(i) + ' H3.2', img)

                    velocity = get_velocity(xml_tracks[i][j], xml_tracks[i][j + 1])
                    cor_x = ((xml_tracks[i][j][0] + xml_tracks[i][j + 1][0]) / 2)
                    cor_y = ((xml_tracks[i][j][1] + xml_tracks[i][j + 1][1]) / 2)
                    data.append([cor_x, cor_y, velocity])
                    c32 += 1
                    # print(str(i) + ' h32 ----- ' + str(x1) + ' ' + str(y1) + ' ++++++ ' + str(x2) + ' ' + str(y2))
                    h3.append(i)

    # print('H1.1 = ' + str(c11))
    # print('H1 = ' + str(c1))
    # print('H1.2 = ' + str(c12))
    #
    # print('H2.1 = ' + str(c21))
    # print('H2 = ' + str(c2))
    # print('H2.2 = ' + str(c22))
    #
    # print('H3.1 = ' + str(c31))
    # print('H3 = ' + str(c3))
    # print('H3.2 = ' + str(c32))

    print('CHECK:')
    print(h1)
    print(h2)
    print(h3)
    #
    # print(data)
    return data
# ----------------------------------------------------------------------------------------------------
def get_results_video(xml_tracks,count):
    # zmenene hodnoty podla ph
    # kontrolovat tracky iba v polovici
    global pixel_size

    # rovnice jednotlivych hranic
    h1a = 3
    h1b = -3
    h1c = -711

    h2a = 0
    h2b = 3
    h2c = -350

    h3a = 3
    h3b = -3
    h3c = -192

    # pole priesecnikov (oblast) pre jednotlive hranice
    # z kazdej strany
    count = count
    h1cs = _get_intersections(h1c, count)
    h2cs = _get_intersections(h2c, count)
    h2cs.reverse()
    h3cs = _get_intersections(h3c, count)

    inf_h1 =[]
    inf_h2 = []
    inf_h3 = []

    dataH1 = []
    dataH2 = []
    dataH3 = []

    for i in range(len(xml_tracks)):
        temp_trac_velocity1 = []
        temp_trac_velocity2 = []
        temp_trac_velocity3 = []
        for j in range(len(xml_tracks[i])-1):
            x1 = xml_tracks[i][j][0]
            y1 = xml_tracks[i][j][1]
            x2 = xml_tracks[i][j + 1][0]
            y2 = xml_tracks[i][j + 1][1]

            # kontrola prvej hranice -------------------------------------------------------------------
            # cez vsetky priesecniky oblasti
            for p in range(len(h1cs)):
                if(_up_line(h1a,h1b,h1cs[p],x1,y1) == True and _up_line(h1a,h1b,h1cs[p],x2,y2) == False):
                    # print('bod presiel 1 hranicu' + str(i))
                    # print(xml_tracks[i][j])
                    # print(xml_tracks[i][j + 1])
                    velocity = get_velocity(xml_tracks[i][j], xml_tracks[i][j + 1])
                    temp_trac_velocity1.append(velocity)
                    if (inf_h1.count(i) == 0):
                        inf_h1.append(i)


            #KONTROLA DRUHEJ HRANICE------------------------------------------------------------------------------
            for p in range(len(h2cs)):
                if(_up_line(h2a,h2b,h2cs[p],x1,y1) == False and _up_line(h2a,h2b,h2cs[p],x2,y2) == True):
                    if (x1 > (600 * pixel_size)):
                        # print('bod presiel 2 hranicu' + str(i))
                        # print(xml_tracks[i][j])
                        # print(xml_tracks[i][j + 1])
                        velocity = get_velocity(xml_tracks[i][j], xml_tracks[i][j + 1])
                        temp_trac_velocity2.append(velocity)
                        if (inf_h2.count(i) == 0):
                            inf_h2.append(i)

            # KONTROLA TRETEJ HRANICE ------------------------------------------------------------------------------------
            for p in range(len(h3cs)):
                if(_up_line(h3a,h3b,h3cs[p],x1,y1) == True and _up_line(h3a,h3b,h3cs[p],x2,y2) == False):
                    if (x1 > (600 * pixel_size)):
                        # print('bod presiel 3 hranicu' + str(i))
                        # print(xml_tracks[i][j])
                        # print(xml_tracks[i][j + 1])
                        velocity = get_velocity(xml_tracks[i][j], xml_tracks[i][j + 1])
                        temp_trac_velocity3.append(velocity)
                        if (inf_h3.count(i) == 0):
                            inf_h3.append(i)

        if(len(temp_trac_velocity1) > 0):
            avg_velocity = sum(temp_trac_velocity1)/len(temp_trac_velocity1)
            dataH1.append(avg_velocity)
            # print('avg za oblast 1 = '+ str(avg_velocity))

        if (len(temp_trac_velocity2) > 0):
            avg_velocity = sum(temp_trac_velocity2) / len(temp_trac_velocity2)
            dataH2.append(avg_velocity)
            # print('avg za oblast 2 = ' + str(avg_velocity))

        if (len(temp_trac_velocity3) > 0):
            avg_velocity = sum(temp_trac_velocity3) / len(temp_trac_velocity3)
            dataH3.append(avg_velocity)
            # print('avg za oblast 3 = ' + str(avg_velocity))


    print(inf_h1)
    print(inf_h2)
    print(inf_h3)
    return dataH1,dataH2,dataH3
# ----------------------------------------

def _up_line(la, lb, lc, x, y):

    eq = la*x+lb*y +lc
    if(eq >= 0):
        return True
    else:
        return False
#     -------------------------------------------------------------------------------------------
def write_results_to_file(file_name,results):
    file = open(file_name,'w')
    file.write('x' + '\t' + 'y' + '\t' + 'v' + '\n')
    for i in range(len(results)):
        file.write(str(results[i][0])+'\t'+str(results[i][1])+'\t'+str(results[i][2])+'\n')
    file.close()

def write_velocity_to_file(file_name,results):
    file = open(file_name,'w')

    for i in range(len(results)):
        file.write('\n'+'v'+str(i)+'\n')
        for j in range(len(results[i])):
            file.write(str(results[i][j])+'\n')
    file.close()
# ----------------------------------------------------------------------------------------
def draw_area_line_video(img_video_tracks,size):
    h11 = [772,61]
    h12 = [880, 169]

    h21 = [772, 350]
    h22 = [880, 350]

    h31 = [772, 580]
    h32 = [880, 688]

    cv2.line(img_video_tracks, (int(h11[0])+ size, int(h11[1] )), (int(h12[0] ), int(h12[1] )- size), (0, 255, 255), 1)
    cv2.line(img_video_tracks, (int(h11[0] ), int(h11[1] )),(int(h12[0] ), int(h12[1] )), (0, 255, 255), 1)
    cv2.line(img_video_tracks, (int(h11[0] )- size, int(h11[1] )),(int(h12[0] ), int(h12[1] )+ size), (0, 255, 255), 1)

    cv2.line(img_video_tracks, (int(h21[0] ), int(h21[1] )+ size), (int(h22[0] ), int(h22[1] )+ size), (0, 255, 255), 1)
    cv2.line(img_video_tracks, (int(h21[0] ), int(h21[1] )),(int(h22[0] ), int(h22[1] )), (0, 255, 255), 1)
    cv2.line(img_video_tracks, (int(h21[0] ), int(h21[1] )- size),(int(h22[0] ), int(h22[1] )- size), (0, 255, 255), 1)

    cv2.line(img_video_tracks, (int(h31[0]  )+ size, int(h31[1] )),(int(h32[0] ), int(h32[1]  )- size), (0, 255, 255), 1)
    cv2.line(img_video_tracks, (int(h31[0] ), int(h31[1] )), (int(h32[0] ), int(h32[1] )), (0, 255, 255), 1)
    cv2.line(img_video_tracks, (int(h31[0] )- size, int(h31[1] )), (int(h32[0] ), int(h32[1] )+ size), (0, 255, 255), 1)



    return img_video_tracks
# -------------------------------------------------------------------------------------------------------------
def draw_area_line_simulation(img_sim_tracks, size):


    cv2.line(img_sim_tracks, (int(67+size)*3, 195*3), (95*3, int(167+size)*3), (0, 255, 255), 1)
    cv2.line(img_sim_tracks, (67*3, 195*3), (95*3, 167*3), (0, 255, 255), 1)
    cv2.line(img_sim_tracks, (int(67-size)*3, 195*3), (95*3, int(167-size)*3), (0, 255, 255), 1)

    cv2.line(img_sim_tracks, (67*3, (100+size)*3), (95*3, (100+size)*3), (0, 255, 255), 1)
    cv2.line(img_sim_tracks, (67*3, 100*3), (95*3, 100*3), (0, 255, 255), 1)
    cv2.line(img_sim_tracks, (67*3, (100-size)*3), (95*3, (100-size)*3), (0, 255, 255), 1)
    # H1
    cv2.line(img_sim_tracks, ((95+size)*3, 5*3), (67*3, (32+size)*3), (0, 255, 255), 1)
    cv2.line(img_sim_tracks, (95*3, 5*3), (67*3, 32*3), (0, 255, 255), 1)
    cv2.line(img_sim_tracks, ((95-size)*3, 5*3), (67*3, (32-size)*3), (0, 255, 255), 1)

    return img_sim_tracks
# ///////////////////////////////////////////////////////////////////////////////////////////////////////

format = input('Would you like to process data from video or simulation? [v/s] ')
if(format == 'v'):
# =====================================NACITANIE TRACKOV Z VIDEA A ICH VYKRESLENIE SPOLU S CIARAMI
    file_name = input('File name: [nova_detekcia.xml]')
    parsed_tracks = parse_xml_tracks('output_tracking\/'+file_name)

    pixel_size = 1/3
    # p = input('Pixel size: ')
    # pp = p.split('/')
    # if(len(pp) > 1):
    #     # print(pp[0])
    #     # print(pp[1])
    #     pixel_size = int(pp[0])/int(pp[1])
    # else:
    #     # print(p)
    #     pixel_size = int(p)
    # # print(pixel_size)
    count = 20
    # resultsH = get_results_video(parsed_tracks,count)
    # results = get_results_video_check(parsed_tracks)
    # file_export = input('File name for export: ')
    # write_velocity_to_file('output_velocity\/'+file_export+'.txt',resultsH)



    show = input('Would you like to show and save img of final tracks?')
    if (show == 'y' or show == 'yes' or show == 'Y' or show == 'YES'):
        name = input('File name:')
        resolution_x = 1280
        resolution_y = 720
        img_video_tracks = np.zeros((int(resolution_y), int(resolution_x), 3), np.uint8)
        img_video_tracks = draw_tracks(parsed_tracks,img_video_tracks)

        # img_video_tracks = draw_area_line_video(img_video_tracks,count)

        cv2.imwrite('img\/' + name + '.png', img_video_tracks)
        cv2.imshow('Tracks from video', img_video_tracks)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

elif (format =='s'):
    file_name = input('File name: [simulation_tracks5.xml] ')
    parsed_tracks = parse_xml_tracks('input_velocity\/' + file_name)

    count = 20
    resultsH=get_results_simulation(parsed_tracks,count)
    # results = get_results_simulation_check(parsed_tracks)

    # file_export = input('File name for export: ')
    # write_velocity_to_file('output_velocity\/' + file_export + '.txt', resultsH)

    show = input('Would you like to show and save img of final tracks?')
    if (show == 'y' or show == 'yes' or show == 'Y' or show == 'YES'):
        name = input('File name:')
        resolution_x = 160*3
        resolution_y = 200*3
        img_sim_tracks = np.zeros((resolution_y,resolution_x, 3), np.uint8)
        img_sim_tracks = draw_tracks(parsed_tracks, img_sim_tracks)
        # img_sim_tracks = draw_area_line_simulation(img_sim_tracks,20)

        cv2.imwrite('img\/' + name + '.png', img_sim_tracks)
        cv2.imshow('Tracks from simulation', img_sim_tracks)
        cv2.waitKey(0)
        cv2.destroyAllWindows()