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

def parse_xml_anastroj_v2(file):
    import xml.etree.ElementTree as ET
    tree = ET.parse(file)
    root = tree.getroot()
    number = -1
    from math import floor
    tracks = []
    src_names = []
    mat = []
    for image in root.find('images').findall('image'):  # image in images
        print(str(image.find('src').text))
        src_names.append(image.find('src').text)
        number += 1
        for bb in image.find('boundingboxes').findall('boundingbox'):
            x_left = int(bb.find('x_left_top').text)
            y_left = int(bb.find('y_left_top').text)
            width = int(bb.find('width').text)
            height = int(bb.find('height').text)
            x = int((x_left + floor(width / 2)))
            y = int((y_left + floor(height / 2)))
            track_id = -1

            class_name = bb.find('class_name')
            if class_name is not None:
                track_id_tag = class_name.find('track_id')
                if track_id_tag is not None and track_id_tag.text is not None:
                    print(track_id_tag.text)
                    track_id = int(track_id_tag.text)

            if (track_id != -1):
                if (str(track_id) != 'false'):
                    if (track_id >= len(tracks)):
                        # for pre tolko, kolko ich treba este pridat
                        print('TI ', (track_id))
                        print('tracks: ', int(len(tracks)) + 1)
                        count_to_add = int(track_id) - int(len(tracks)) + 1
                        for i in range(count_to_add):
                            track = []
                            tracks.append(track)
                    tracks[int(track_id)].append([x, y, 1, number, width, height])

            if (number >= len(mat)):
                # for pre tolko, kolko ich treba este pridat
                count_to_add = int(number) - int(len(mat)) + 1
                for i in range(count_to_add):
                    frame = []
                    mat.append(frame)
            if (track_id == -1):
                mat[int(number)].append([x, y, 0, number, width, height])
            else:
                mat[int(number)].append([x, y, 1, number, width, height])
                # number += 1

    print('--- *************** TRACKS: ************************** ---')
    print(tracks)
    print('- - - - - - - - - - - - - - - - - - - - - - - - ')
    print('--- ***************** MATRIX: ************************ ---')
    print(mat)
    print('- - - - - - - - - - - - - - - - - - - - - - - - ')
    print(src_names)
    return tracks, mat, src_names


def parse_xml_anastroj2_2(file):
    print('Parsing: ' + file)
    import xml.etree.ElementTree as ET
    tree = ET.parse(file)
    root = tree.getroot()
    number = -1
    from math import floor
    tracks = []
    src_names = []
    mat = []
    for image in root.find('images').findall('image'):  # image in images
        print(str(image.find('src').text))
        src_names.append(image.find('src').text)
        for child3 in image:  # boundingboxes in image
            if (child3.tag == 'boundingboxes'):
                boundingboxes = child3
                number += 1
                for boundingbox in boundingboxes.iter('boundingbox'):
                    x_left = int(boundingbox.find('x_left_top').text)
                    y_left = int(boundingbox.find('y_left_top').text)
                    width = int(boundingbox.find('width').text)
                    height = int(boundingbox.find('height').text)
                    x = int((x_left + floor(width / 2)))
                    y = int((y_left + floor(height / 2)))
                    track_id = -1

                    class_name = boundingbox.find('class_name')
                    if class_name is not None:
                        track_id_tag = class_name.find('track_id')
                        if track_id_tag is not None:
                            if track_id_tag.text == '1-50\video2359_0001.tiff':
                                pass
                            print(track_id_tag.text)
                            track_id = int(track_id_tag.text)

                    if (track_id != -1):
                        if (str(track_id) != 'false'):
                            if (track_id >= len(tracks)):
                                # for pre tolko, kolko ich treba este pridat
                                print('TI ', (track_id))
                                print('tracks: ', int(len(tracks)) + 1)
                                count_to_add = int(track_id) - int(len(tracks)) + 1
                                for i in range(count_to_add):
                                    track = []
                                    tracks.append(track)
                            tracks[int(track_id)].append([x, y, 1, number, width, height])

                    if (number >= len(mat)):
                        # for pre tolko, kolko ich treba este pridat
                        count_to_add = int(number) - int(len(mat)) + 1
                        for i in range(count_to_add):
                            frame = []
                            mat.append(frame)
                    if (track_id == -1):
                        mat[int(number)].append([x, y, 0, number, width, height])
                    else:
                        mat[int(number)].append([x, y, 1, number, width, height])
                        # number += 1



    print('--- *************** TRACKS: ************************** ---')
    print(tracks)
    print('- - - - - - - - - - - - - - - - - - - - - - - - ')
    print('--- ***************** MATRIX: ************************ ---')
    print(mat)
    print('- - - - - - - - - - - - - - - - - - - - - - - - ')
    print(src_names)
    return tracks, mat, src_names

def parse_xml_anastroj2(file):
    print('Parsing: ' + file)
    import xml.etree.ElementTree as ET
    tree = ET.parse(file)
    root = tree.getroot()
    number = -1
    from math import floor
    tracks = []
    src_names = []
    mat = []
    for image in root.find('images').findall('image'):  # image in images
        print(str(image.find('src').text))
        src_names.append(image.find('src').text)
        for child3 in image:  # boundingboxes in image
            if (child3.tag == 'boundingboxes'):
                boundingboxes = child3
                number += 1
                for boundingbox in boundingboxes.iter('boundingbox'):
                    x_left = int(boundingbox.find('x_left_top').text)
                    y_left = int(boundingbox.find('y_left_top').text)
                    width = int(boundingbox.find('width').text)
                    height = int(boundingbox.find('height').text)
                    x = int((x_left + floor(width / 2)))
                    y = int((y_left + floor(height / 2)))
                    track_id = -1

                    for att in boundingbox.iter('attribute'):
                        if (att.text != None):
                            track_id = (att.text)

                    if (track_id != -1):
                        if (str(track_id) != 'false'):
                            if (track_id >= str(len(tracks))):
                                # for pre tolko, kolko ich treba este pridat
                                print('TI ', (track_id))
                                print('tracks: ', int(len(tracks)) + 1)
                                count_to_add = int(track_id) - int(len(tracks)) + 1
                                for i in range(count_to_add):
                                    track = []
                                    tracks.append(track)
                            tracks[int(track_id)].append([x, y, 1, number, width, height])

                    if (number >= len(mat)):
                        # for pre tolko, kolko ich treba este pridat
                        count_to_add = int(number) - int(len(mat)) + 1
                        for i in range(count_to_add):
                            frame = []
                            mat.append(frame)
                    if (track_id == -1):
                        mat[int(number)].append([x, y, 0, number, width, height])
                    else:
                        mat[int(number)].append([x, y, 1, number, width, height])
                        # number += 1



    print('--- *************** TRACKS: ************************** ---')
    print(tracks)
    print('- - - - - - - - - - - - - - - - - - - - - - - - ')
    print('--- ***************** MATRIX: ************************ ---')
    print(mat)
    print('- - - - - - - - - - - - - - - - - - - - - - - - ')
    print(src_names)
    return tracks, mat, src_names
# ------------------------------------------------------------------------------------------
def parse_xml_anastroj(file):
    print('Parsing: ' + file)
    import xml.etree.ElementTree as ET
    tree = ET.parse(file)
    root = tree.getroot()
    number = -1
    from math import floor
    tracks = []
    src_names = []
    mat = []

    for child in root: # images in data
        for child2 in child: # image in images
            print(str(child2.find('src').text))
            src_names.append(child2.find('src').text)
            for child3 in child2: # boundingboxes in image
                if (child3.tag == 'boundingboxes'):
                    boundingboxes = child3
                    number += 1
                    for boundingbox in boundingboxes.iter('boundingbox'):
                        x_left = int(boundingbox.find('x_left_top').text)
                        y_left = int(boundingbox.find('y_left_top').text)
                        width = int(boundingbox.find('width').text)
                        height = int(boundingbox.find('height').text)
                        x = int((x_left + floor(width / 2)))
                        y = int((y_left + floor(height/2)))
                        track_id = -1

                        for att in boundingbox.iter('attribute'):
                            if (att.text != None):
                                track_id = (att.text)

                        if (track_id != -1):
                            if (str(track_id) != 'false'):
                                if (int(track_id) >= len(tracks)):
                                    # for pre tolko, kolko ich treba este pridat
                                    print('TI ', (track_id))
                                    print('tracks: ', int(len(tracks)) + 1)
                                    count_to_add = int(track_id) - int(len(tracks)) + 1
                                    for i in range(count_to_add):
                                        track = []
                                        tracks.append(track)
                                tracks[int(track_id)].append([x, y, 1, number, width, height])

                        if (number >= len(mat)):
                            # for pre tolko, kolko ich treba este pridat
                            count_to_add = int(number) - int(len(mat)) + 1
                            for i in range(count_to_add):
                                frame = []
                                mat.append(frame)
                        if (track_id == -1):
                            mat[int(number)].append([x, y, 0, number, width, height])
                        else:
                            mat[int(number)].append([x, y, 1, number, width, height])
                    #number += 1


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
    import xml.etree.ElementTree as element
    from io import BytesIO
    from math import floor

    document = element.Element('outer')
    node = element.SubElement(document, 'inner')
    et = element.ElementTree(document)
    f = BytesIO()
    et.write(f, encoding='utf-8', xml_declaration=True)
    print(f.getvalue())  # your XML file, encoded as UTF-8
    #root = element.SubElement(et,"data")



    root = element.Element("data")
    images = element.SubElement(root, "images")

    for x in range(len(mat)):
        image = element.SubElement(images,"image")

        if (len(src_names) != 0):
            src = element.SubElement(image, "src")
            src.text = str(src_names[x])

        boundingboxes = element.SubElement(image,"boundingboxes")
        for y in range(len(mat[x])):
            boundingbox = element.SubElement(boundingboxes,"boundingbox")
            x_left_top = element.SubElement(boundingbox, "x_left_top")
            x_left_top.text = str(mat[x][y][0]- floor(mat[x][y][4]/2))

            y_left_top = element.SubElement(boundingbox, "y_left_top")
            y_left_top.text = str(mat[x][y][1] - floor(mat[x][y][5]/2))

            width = element.SubElement(boundingbox, "width")
            width.text = str(mat[x][y][4])

            height = element.SubElement(boundingbox, "height")
            height.text = str(mat[x][y][5])
            class_name = element.SubElement(boundingbox, "class", {"name": 'bunka'})
            attribute = element.SubElement(class_name, "attribute", {"name":'track_id'})

            for n in range(len(tracks)):
                if (mat[x][y] in tracks[n]):
                    attribute.text = str(n)

    tree = element.ElementTree(root)
    tree.write('output_tracking\/' + file_name + '.xml')
    print('Done')
