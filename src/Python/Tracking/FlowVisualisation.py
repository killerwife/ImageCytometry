import XMLRead
import cv2

firstVideo = False

if firstVideo:
    ANNOTATIONS_FILE_NAME = 'tracks_1_300.xml'
    background = 'D:\\BigData\\cellinfluid\\backgrounds\\background.png'
else:
    ANNOTATIONS_FILE_NAME = 'deformabilityAnnotations.xml'
    background = 'D:\\BigData\\cellinfluid\\backgrounds\\backgroundDeform.png'

fileName = 'C:\\GitHubCode\\phd\\ImageCytometry\\src\\XML\\' + ANNOTATIONS_FILE_NAME
annotatedData = []
XMLRead.readXML(fileName, annotatedData)
tracks = XMLRead.initTracks(annotatedData)
image = cv2.imread(background)
for key, track in tracks.items():
    for i in range(len(track.boundingBoxes) - 1):
        cv2.arrowedLine(image, (track.boundingBoxes[i].x + int(track.boundingBoxes[i].width / 2),
                                track.boundingBoxes[i].y + int(track.boundingBoxes[i].height / 2)),
                        (track.boundingBoxes[i + 1].x + int(track.boundingBoxes[i + 1].width / 2),
                         track.boundingBoxes[i + 1].y + int(track.boundingBoxes[i + 1].height / 2)), (0, 0, 255), 1)

cv2.imwrite('Image.png', image)
cv2.waitKey()


