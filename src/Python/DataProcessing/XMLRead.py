import xml.etree.ElementTree as ET
from xml.dom import minidom


class BoundingBox(object):
    def __init__(self):
        self.x = 0
        self.y = 0
        self.width = 0
        self.height = 0
        self.className = "bunka"
        self.trackId = -1


class Image(object):
    def __init__(self):
        self.boundingBoxes = []
        self.filename = ""


def readXML(fileName, imageData):
    tree = ET.parse(fileName)
    root = tree.getroot()
    version = 1
    metadata = root.find('metadata')
    if metadata is not None:
        versionXML = metadata.find('version_major')
        if versionXML is not None:
            version = int(versionXML.text)

    for child in root.find('images'):
        image = Image()
        image.filename = child.find('src').text
        boundingBoxes = child.find('boundingboxes')
        for boundingBoxXML in boundingBoxes.findall('boundingbox'):
            boundingBox = BoundingBox()
            boundingBox.x = int(boundingBoxXML.find('x_left_top').text)
            boundingBox.y = int(boundingBoxXML.find('y_left_top').text)
            boundingBox.width = int(boundingBoxXML.find('width').text)
            boundingBox.height = int(boundingBoxXML.find('height').text)
            if version == 1:
                className = boundingBoxXML.find('class')
                if className is None:
                    boundingBox.className = 'bunka'
                else:
                    boundingBox.className = className.get('name')
                    for attribute in className.findall('attribute'):
                        if attribute.get('name') == 'track_id':
                            boundingBox.trackId = int(attribute.text)
                            break
            elif version == 2:
                attributes = boundingBoxXML.find('class_name')
                if attributes is None:
                    boundingBox.className = 'bunka'
                else:
                    className = attributes.find('project_id')
                    trackId = attributes.find('track_id')
                    if className is None:
                        boundingBox.className = 'bunka'
                    else:
                        boundingBox.className = className.text
                    if trackId is None:
                        boundingBox.trackId = -1
                    else:
                        if trackId.text is None:
                            boundingBox.trackId = -1
                        else:
                            boundingBox.trackId = int(trackId.text)

            image.boundingBoxes.append(boundingBox)
        imageData.append(image)


fileNameFirst = '../../XML/track_1_51_151_200.xml'
fileNameSecond = '../../XML/trasy_51_151_xml2.xml'
fileNameOutput = '../../XML/tracks_1_200.xml'
imageDataFirst = []
imageDataSecond = []
readXML(fileNameFirst, imageDataFirst)
readXML(fileNameSecond, imageDataSecond)

def replaceTrackIds(trackPairs, imageData):
    for image in imageData:
        for boundingBox in image.boundingBoxes:
            resultingTrackId = trackPairs.get(boundingBox.trackId)
            if resultingTrackId is not None and resultingTrackId != -1:
                boundingBox.trackId = trackPairs.get(boundingBox.trackId)

def connectTracks(imageDataFirst, imageDataSecond):
    jointImages = []
    for imageOne in imageDataFirst:
        for imageTwo in imageDataSecond:
            if imageOne.filename == imageTwo.filename:
                if imageOne.boundingBoxes[0].trackId == -1 and imageTwo.boundingBoxes[0].trackId == -1:
                    jointImages.append(imageTwo)  # if no tracks in either, eliminate any one of them
                    break
                if imageOne.boundingBoxes[0].trackId == -1:
                    jointImages.append(imageOne)
                    break
                elif imageTwo.boundingBoxes[0].trackId == -1:
                    jointImages.append(imageTwo)
                    break
                trackList = {}
                trackListReverse = {}
                for boundingBoxOne in imageOne.boundingBoxes:
                    for boundingBoxTwo in imageTwo.boundingBoxes:
                        if boundingBoxOne.x == boundingBoxTwo.x and boundingBoxOne.y == boundingBoxTwo.y and \
                               boundingBoxOne.width == boundingBoxTwo.width and boundingBoxOne.height == boundingBoxTwo.height:
                            trackList[boundingBoxTwo.trackId] = boundingBoxOne.trackId
                            trackListReverse[boundingBoxOne.trackId] = boundingBoxTwo.trackId
                            break
                if int(imageOne.filename[-9:-5]) == 51:
                    replaceTrackIds(trackList, imageDataSecond)
                else:
                    replaceTrackIds(trackListReverse, imageDataFirst)
                jointImages.append(imageTwo)
                break
    mergedImageData = imageDataFirst + imageDataSecond
    for image in jointImages:
        mergedImageData.remove(image)
    return mergedImageData


mergedImageData = connectTracks(imageDataFirst, imageDataSecond)
def getKey(filename):
    return int(filename[-9:-5])

mergedImageData.sort(key= lambda x: getKey(x.filename))

def writeXML(imageData, fileNameOutput):
    root = ET.Element('data')
    metadata = ET.SubElement(root, 'metadata')
    ET.SubElement(metadata, 'data_id').text = 'sa_dataset'
    ET.SubElement(metadata, 'parent')
    ET.SubElement(metadata, 'version_major').text = '2'
    ET.SubElement(metadata, 'xml_sid')
    ET.SubElement(metadata, 'description').text = 'Anotacny nastroj v1.02'
    images = ET.SubElement(root, 'images')
    for image in imageData:
        imageXML = ET.SubElement(images, 'image')
        ET.SubElement(imageXML, 'src').text = image.filename
        boundingBoxesXML = ET.SubElement(imageXML, 'boundingboxes')
        for boundingBox in image.boundingBoxes:
            boundingBoxXML = ET.SubElement(boundingBoxesXML, 'boundingbox')
            ET.SubElement(boundingBoxXML, 'x_left_top').text = str(boundingBox.x)
            ET.SubElement(boundingBoxXML, 'y_left_top').text = str(boundingBox.y)
            ET.SubElement(boundingBoxXML, 'width').text = str(boundingBox.width)
            ET.SubElement(boundingBoxXML, 'height').text = str(boundingBox.height)
            classNameXML = ET.SubElement(boundingBoxXML, 'class_name')
            ET.SubElement(classNameXML, 'project_id').text = boundingBox.className
            if boundingBox.trackId != -1:
                ET.SubElement(classNameXML, 'track_id').text = str(boundingBox.trackId)
            else:
                ET.SubElement(classNameXML, 'track_id').text


    xmlstr = minidom.parseString(ET.tostring(root, encoding="UTF-8")).toprettyxml(indent="  ", encoding='utf-8')
    with open(fileNameOutput, "wb") as f:
        f.write(xmlstr)


writeXML(mergedImageData, fileNameOutput)
# for data in imageData:
#     print(data.filename)
#     for boundingBox in data.boundingBoxes:
#         print(boundingBox.trackId)

