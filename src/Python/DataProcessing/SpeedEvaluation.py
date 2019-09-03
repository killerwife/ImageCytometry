import XMLRead
import math


class TrackedBoundingBox(XMLRead.BoundingBox):
    def __init__(self):
        XMLRead.BoundingBox.__init__(self)
        self.frameId = 0


class Track(object):
    def __init__(self):
        self.trackId = -1
        self.boundingBoxes = []


fileNamePredicted = 'D:\\BigData\\cellinfluid\\bunkyObrazkyTiff\\tracks_1_200.xml'
annotatedData = []
XMLRead.readXML(fileNamePredicted, annotatedData)


def initTracks(annotatedData):
    i = 0
    tracks = {}
    for image in annotatedData:
        for boundingBox in image.boundingBoxes:
            if boundingBox.trackId != -1:
                trackedBoundingBox = TrackedBoundingBox()
                trackedBoundingBox.trackId = boundingBox.trackId
                trackedBoundingBox.x = boundingBox.x
                trackedBoundingBox.y = boundingBox.y
                trackedBoundingBox.height = boundingBox.height
                trackedBoundingBox.width = boundingBox.width
                trackedBoundingBox.frameId = i
                if boundingBox.trackId not in tracks:
                    track = Track()
                    track.trackId = boundingBox.trackId
                    tracks[boundingBox.trackId] = track

                tracks[boundingBox.trackId].boundingBoxes.append(trackedBoundingBox)
        i += 1
    return tracks


tracks = initTracks(annotatedData)


def computeSpeeds(tracks):
    for key, track in tracks.items():
        speeds = 0
        i = 0
        for trackedBoundingBox in track.boundingBoxes:
            if i == 0:
                i += 1
                continue
            firstBoundingBox = track.boundingBoxes[i - 1]
            if firstBoundingBox.frameId == trackedBoundingBox.frameId:
                print('bla')
            speeds += math.sqrt((trackedBoundingBox.x - firstBoundingBox.x)**2 + (trackedBoundingBox.y - firstBoundingBox.y)**2) / abs(firstBoundingBox.frameId - trackedBoundingBox.frameId)
            i += 1
        print('Track ID: ' + str(track.trackId) + ' has speed:' + str(speeds / i))


computeSpeeds(tracks)