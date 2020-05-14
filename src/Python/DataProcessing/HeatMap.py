import XMLRead
import numpy as np
import cv2

PATH_TO_ANNOTATED_DATA = 'C:\\GitHubCode\\phd\\ImageCytometry\\src\\XML\\' + 'deformabilityAnnotations.xml'
PATH_TO_BACKGROUND = 'D:\\BigData\\cellinfluid\\backgrounds\\backgroundDeform.png'
annotatedData = []
XMLRead.readXML(PATH_TO_ANNOTATED_DATA, annotatedData)
heatMap = np.zeros((720, 1280))

for image in annotatedData:
    for boundingBox in image.boundingBoxes:
        for i in range(boundingBox.x, boundingBox.x + boundingBox.width):
            for k in range(boundingBox.y, boundingBox.y + boundingBox.height):
                heatMap[k][i] += 1

image = cv2.imread(PATH_TO_BACKGROUND)
# cv2.imshow('Bla', image)
# cv2.waitKey()
for i in range(720):
    for k in range(1280):
        value = heatMap[i][k]
        if value > 75:
            image[i][k] = [0, 0, 255]  # red
        elif value > 65:
            image[i][k] = [0, 75, 255]  # orange-ish red
        elif value > 55:
            image[i][k] = [0, 165, 255]  # orange
        elif value > 40:
            image[i][k] = [0, 255, 255]  # yellow
        elif value > 30:
            image[i][k] = [63, 255, 173]  # green
        elif value > 20:
            image[i][k] = [127, 255, 0]  # green/blue
        elif value > 10:
            image[i][k] = [255, 150, 0]  # blueish
        elif value > 0:
            image[i][k] = [255, 0, 0]  # blueish

cv2.imwrite('HeatMap.png', image)
cv2.waitKey()
