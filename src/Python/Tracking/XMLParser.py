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
