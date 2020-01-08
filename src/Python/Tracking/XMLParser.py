import os, sys; sys.path.append(os.path.dirname(os.path.realpath(__file__)))
import xml.etree.ElementTree as ET
from math import floor
import XMLRead
from xml.dom import minidom
from io import BytesIO

def parse_xml(file):
    # naplnenie matice mat suradnicami bodov podla framov. Ak je zadama aj matrix- naplni sa "pomocna matica" (i,j) obsahuju cisla framov, v ktorych sa nachadzaju body so suradnicami i,j
    print('Parsing: ' + file)
    from xml.dom import minidom
    xmldoc = minidom.parse(file)
    frames = xmldoc.getElementsByTagName('frame')
    particles = xmldoc.getElementsByTagName('particle')
    print('\tframes: ' + str(len(frames)))
    # print(len(particles))
    range_from = 0
    mat =[]
    total_count = 0

    for f in frames:
        frame = []
        number = int(f.attributes['number'].value)
        count = int(f.attributes['particlesCount'].value)
        total_count += count
        for p in range(range_from, (range_from + count)):
            # print(particles[p].attributes['number'].value+', '+particles[p].attributes['x'].value+', '+particles[p].attributes['y'].value)
            # par_number = int(particles[p].attributes['number'].value)
            par_x = int(particles[p].attributes['x'].value)
            par_y = int(particles[p].attributes['y'].value)
            # frame.append([par_number,par_x, par_y])

            frame.append([par_x, par_y, 0, number, 26, 26])
            # if (fill_matrix):
            #     matrix[par_x][par_y].append(int(number))
            #     total_count += 1

        mat.append(frame)
        range_from = range_from + int(count)
    print('\tparticles: ' + str(total_count) )
    print('\tdone')
    return mat


def parseXMLData(filename, loadTracks):
    annotatedData = []
    XMLRead.readXML(filename, annotatedData)
    number = -1
    tracks = []
    src_names = []
    mat = []
    for image in annotatedData: # image in images
        print(image.filename)
        src_names.append(image.filename)
        number += 1
        for boundingBox in image.boundingBoxes: # boundingboxes in image
            x_left = int(boundingBox.x)
            y_left = int(boundingBox.y)
            width = int(boundingBox.width)
            height = int(boundingBox.height)
            x = int((x_left + floor(width / 2)))
            y = int((y_left + floor(height/2)))
            track_id = boundingBox.trackId

            if track_id != -1:
                if track_id >= len(tracks):
                    # for pre tolko, kolko ich treba este pridat
                    print('TI ', track_id)
                    print('tracks: ', int(len(tracks)) + 1)
                    count_to_add = track_id - int(len(tracks)) + 1
                    for i in range(count_to_add):
                        track = []
                        tracks.append(track)
                tracks[track_id].append([x, y, 1, number, width, height])

            if number >= len(mat):
                # for pre tolko, kolko ich treba este pridat
                count_to_add = int(number) - int(len(mat)) + 1
                for i in range(count_to_add):
                    frame = []
                    mat.append(frame)
            if track_id == -1 or loadTracks == False:
                mat[int(number)].append([x, y, 0, number, width, height])
            else:
                mat[int(number)].append([x, y, 1, number, width, height])

    print('--- *************** TRACKS: ************************** ---')
    print(tracks)
    print('- - - - - - - - - - - - - - - - - - - - - - - - ')
    print('--- ***************** MATRIX: ************************ ---')
    print(mat)
    print('- - - - - - - - - - - - - - - - - - - - - - - - ')
    print(src_names)
    return tracks, mat, src_names

# TODO pridat nezaradene body
# ------------------------------------------------------------------------------------------
def save_as_anastroj_file(mat, tracks, src_names, file_name):
    print('save file as anastroj: ', file_name)
    imageData = []

    for x in range(len(mat)):
        image = XMLRead.Image()

        if (len(src_names) != 0):
            image.filename = str(src_names[x])

        for y in range(len(mat[x])):
            boundingBox = XMLRead.BoundingBox()
            boundingBox.x = mat[x][y][0]- floor(mat[x][y][4]/2)
            boundingBox.y = mat[x][y][1] - floor(mat[x][y][5]/2)
            boundingBox.width = mat[x][y][4]
            boundingBox.height = mat[x][y][5]

            for n in range(len(tracks)):
                if (mat[x][y] in tracks[n]):
                    boundingBox.trackId = n  # determine if we need more than one track per boundingBox
                    break
            image.boundingBoxes.append(boundingBox)
        imageData.append(image)

    XMLRead.writeXML(imageData, file_name)
    print('Done')
