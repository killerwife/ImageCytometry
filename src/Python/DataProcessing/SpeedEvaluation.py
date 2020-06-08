import XMLRead
import math

fileNamePredicted = 'C:\\GitHubCode\\phd\\ImageCytometry\\src\\XML\\deformabilityAnnotations.xml'
annotatedData = []
XMLRead.readXML(fileNamePredicted, annotatedData)

tracks = XMLRead.initTracks(annotatedData)

def computeSpeeds(tracks):
    max = 0
    avg = 0
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
        avg += speeds / i
        if speeds / i > max:
            max = speeds / i
    print('Max speed: ' + str(max))
    print('Avg speed: ' + str(avg / len(tracks)))


computeSpeeds(tracks)
