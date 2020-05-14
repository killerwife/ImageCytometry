import XMLRead
import cv2

firstVideo = False

if firstVideo:
    ANNOTATIONS_FILE_NAME = 'tracks_1_300.xml'
    background = 'D:\\BigData\\cellinfluid\\backgrounds\\background.png'
else:
    ANNOTATIONS_FILE_NAME = 'deformabilityAnnotations.xml'
    background = 'D:\\BigData\\cellinfluid\\backgrounds\\backgroundDeform.png'

XML_FOLDER_PREFIX = 'C:\\GitHubCode\\phd\\ImageCytometry\\src\\XML\\'
fileName = XML_FOLDER_PREFIX + ANNOTATIONS_FILE_NAME
annotatedData = []
trackedData = []
XMLRead.readXML(fileName, annotatedData)
XMLRead.readXML(XML_FOLDER_PREFIX + 'output.xml', trackedData)
annotatedTracks = XMLRead.initTracks(annotatedData)
tracedTracks = XMLRead.initTracks(trackedData)
image = cv2.imread(background)

# visualise not assigned bbs
for imageData in trackedData:
    for boundingBox in imageData.boundingBoxes:
        if boundingBox.trackId == -1:
            for key, value in annotatedTracks.items():
                for i in range(len(value.boundingBoxes)):
                    if value.boundingBoxes[i].x == boundingBox.x and value.boundingBoxes[i].y == boundingBox.y:
                        if i > 0:
                            cv2.line(image, (value.boundingBoxes[i - 1].x, value.boundingBoxes[i - 1].y),
                                     (boundingBox.x, boundingBox.y), [0, 0, 255], 2)
                        if i < len(value.boundingBoxes) - 1:
                            cv2.line(image, (boundingBox.x, boundingBox.y),
                                     (value.boundingBoxes[i + 1].x, value.boundingBoxes[i + 1].y), [0, 0, 255], 2)
                        cv2.circle(image, (boundingBox.x, boundingBox.y), 4, [0, 255, 0], 1)

# data outputs
# boundingBoxCount = 0
# trackCount = 0
# averageWidth = 0
# averageHeight = 0
# averageTrackLength = 0
# uniqueSet = set()
# for imageData in annotatedData:
#     boundingBoxCount += len(imageData.boundingBoxes)
#     for boundingBox in imageData.boundingBoxes:
#         averageWidth += boundingBox.width
#         averageHeight += boundingBox.height
#
# for key, track in annotatedTracks.items():
#     averageTrackLength += len(track.boundingBoxes)
#     uniqueSet.add(track.trackId)
#
# trackCount = len(uniqueSet)
# averageTrackLength /= trackCount
# averageWidth /= boundingBoxCount
# averageHeight /= boundingBoxCount
#
# print('Bounding boxes in dataset: ' + str(boundingBoxCount))
# print('Tracks in dataset: ' + str(trackCount))
# print('Average Width of BB: ' + str(averageWidth) + ' Average Height of BB: ' + str(averageHeight))
# print('Average Track Length: ' + str(averageTrackLength))

cv2.imwrite('Image.png', image)
cv2.waitKey()


