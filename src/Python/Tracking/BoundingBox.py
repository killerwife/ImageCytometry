import math

maxPosDiff = float(0.5)
lowestOverlap = float(0.5)

# TODO: refactor xml bounding box with this bounding box
def evaluateBoundingBox(original, predicted):
    biggerLeft = 0
    biggerRight = 0
    if original.x > predicted.x:
        biggerLeft = original.x
    else:
        biggerLeft = predicted.x
    if original.x + original.width < predicted.x + predicted.width:
        biggerRight = original.x + original.width
    else:
        biggerRight = predicted.x + predicted.width
    overlapWidth = biggerRight - biggerLeft
    if original.y > predicted.y:
        biggerLeft = original.y
    else:
        biggerLeft = predicted.y
    if original.y + original.height < predicted.y + predicted.height:
        biggerRight = original.y + original.height
    else:
        biggerRight = predicted.y + predicted.height
    overlapHeight = biggerRight - biggerLeft
    originalArea = original.width * original.height
    distance = ((original.x + original.width / 2) - (predicted.x + predicted.width / 2))**2 + ((original.y + original.height / 2) - (predicted.y + predicted.height / 2))**2
    distance = math.sqrt(float(distance))
    overlapMaxdist = predicted.width * maxPosDiff
    if distance < overlapMaxdist and overlapWidth > 0 and overlapHeight > 0:
        return float(overlapWidth) * overlapHeight / originalArea, distance / predicted.width
    else:
        return float(0), float(0)

class TrackBoundingBox:

    def __init__(self, x, y, in_track, frame, width, height):
        self.x = x
        self.y = y
        self.in_track = in_track
        self.frame_index = frame
        self.width = width
        self.height = height
        self.true_positive = False
        self.true_positive2 = False

    def __str__(self):
        return "[" + str(self.x) + "," + str(self.y) + "," + str(self.in_track) + "," + str(self.frame_index) + "," + "]"

    def __eq__(self, other):
        overlapRatio, centerDistRatio = evaluateBoundingBox(self, other)
        return overlapRatio > lowestOverlap

    def __ne__(self, other):
        if self == other:
            return False
        return True
