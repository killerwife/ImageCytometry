import tensorflow as tf
import XMLRead
import cv2
import numpy as np
import CellDataReader
import os
import Definitions
from object_detection.utils import dataset_util

class DatasetSegment(object):
    def __init__(self, startExclusion, endExclusion, startBoundary, endBoundary):
        self.startExclusion = startExclusion
        self.endExclusion = endExclusion
        self.startBoundary = startBoundary
        self.endBoundary = endBoundary

class TrackingDataset(object):
    def __init__(self, width, height, x, y, features, response):
        self.width = width
        self.height = height
        self.x = x
        self.y = y
        self.features = features
        self.response = response

class Dataset(object):
    # detection
    def createTFRecord(self, imageData, imagePath):
        # image = cv2.imread(imagePath + imageData.filename, cv2.IMREAD_GRAYSCALE)
        image = cv2.imread(imagePath + imageData.filename, cv2.IMREAD_COLOR)
        filename = imageData.filename  # Filename of the image. Empty if image is not from file
        encoded_image_data = cv2.imencode('.png', image)[1].tostring()
        encoded_image_string_tf = tf.compat.as_bytes(encoded_image_data)
        height = float(image.shape[0])  # Image height
        width = float(image.shape[1])  # Image width
        image_format = b'png'  # b'jpeg' or b'png'

        xmins = []  # List of normalized left x coordinates in bounding box (1 per box)
        xmaxs = []  # List of normalized right x coordinates in bounding box
        # (1 per box)
        ymins = []  # List of normalized top y coordinates in bounding box (1 per box)
        ymaxs = []  # List of normalized bottom y coordinates in bounding box
        # (1 per box)
        classes_text = []  # List of string class name of bounding box (1 per box)
        classes = []  # List of integer class id of bounding box (1 per box)

        for boundingBox in imageData.boundingBoxes:
            xmins.append(float(boundingBox.x) / width)
            xmaxs.append((float(boundingBox.x) + float(boundingBox.width)) / width)
            ymins.append(float(boundingBox.y) / height)
            ymaxs.append((float(boundingBox.y) + float(boundingBox.height)) / height)
            classes_text.append(b'Cell')
            classes.append(1)

        tf_example = tf.train.Example(features=tf.train.Features(feature={
            'image/height': dataset_util.int64_feature(image.shape[0]),
            'image/width': dataset_util.int64_feature(image.shape[1]),
            'image/filename': dataset_util.bytes_feature(str.encode(filename)),
            'image/source_id': dataset_util.bytes_feature(str.encode(filename)),
            'image/encoded': dataset_util.bytes_feature(encoded_image_string_tf),
            'image/format': dataset_util.bytes_feature(image_format),
            'image/object/bbox/xmin': dataset_util.float_list_feature(xmins),
            'image/object/bbox/xmax': dataset_util.float_list_feature(xmaxs),
            'image/object/bbox/ymin': dataset_util.float_list_feature(ymins),
            'image/object/bbox/ymax': dataset_util.float_list_feature(ymaxs),
            'image/object/class/text': dataset_util.bytes_list_feature(classes_text),
            'image/object/class/label': dataset_util.int64_list_feature(classes),
        }))
        return tf_example

    def processXML(self, writers, annotatedData, imageFolders, boundary, index):
        i = -1
        counters = []
        counters.append(0)
        counters.append(0)
        for example in annotatedData:
            i = i + 1
            if i >= boundary.startExclusion and i < boundary.endExclusion:
                continue

            writerIndex = 0
            if i >= boundary.startBoundary and i < boundary.endBoundary:
                writerIndex = 1
            tf_example = self.createTFRecord(example, imageFolders[index])
            writers[writerIndex].write(tf_example.SerializeToString())
            counters[writerIndex] += 1
        print("Inserted images to index " + str(index) + " : " + str(counters[0]) + " " + str(counters[1]))

    def generateTfRecord(self, folderPath, recordName, imageFolders, boundaries, xmlFiles):
        writers = []
        writers.append(tf.python_io.TFRecordWriter(folderPath + 'train' + recordName + '.record'))
        writers.append(tf.python_io.TFRecordWriter(folderPath + 'eval' + recordName + '.record'))

        i = 0
        while i < len(imageFolders):
            annotatedData = []
            filePathImageData = xmlFiles[i]
            XMLRead.readXML(filePathImageData, annotatedData)
            self.processXML(writers, annotatedData, imageFolders, boundaries[i], i)
            i += 1

        writers[0].close()
        writers[1].close()

    # tracking
    def createTrackingTFRecord(self, datavector, width, height, x, y, response):
        tf_example = tf.train.Example(features=tf.train.Features(feature={
            'data/height': dataset_util.int64_feature(height),
            'data/width': dataset_util.int64_feature(width),
            'data/x': dataset_util.int64_feature(x),
            'data/y': dataset_util.int64_feature(y),
            'data/features': dataset_util.float_list_feature(datavector.flatten()),
            'data/response': dataset_util.float_list_feature(response),
        }))
        return tf_example

    def createTrackingDataset(self, folderPath, annotatedData, recordName, flowmatrix):
        tracks = XMLRead.initTracks(annotatedData)
        writers = []
        entries = 0
        writers.append(tf.python_io.TFRecordWriter(folderPath + 'train' + recordName + '.record'))
        writers.append(tf.python_io.TFRecordWriter(folderPath + 'eval' + recordName + '.record'))
        array = np.zeros((60, 60, 5), np.float32)  # index 0 - x velocity index 1 y velocity index 2 other cells

        # velocities = np.zeros((720, 1280, 2), np.float32)
        # for currentKey, currentTrack in tracks.items():
        #     for curBoxIndex in range(len(currentTrack.boundingBoxes)):
        #         boundingBox = currentTrack.boundingBoxes[curBoxIndex]
        #         if curBoxIndex > 0:
        #             curPrevBoundingBox = currentTrack.boundingBoxes[curBoxIndex - 1]
        #             fillingVelocityX = (boundingBox.x - curPrevBoundingBox.x) / float(
        #                 abs(curPrevBoundingBox.frameId - boundingBox.frameId))
        #             fillingVelocityY = (boundingBox.y - curPrevBoundingBox.y) / float(
        #                 abs(curPrevBoundingBox.frameId - boundingBox.frameId))
        #             for indexY in range(boundingBox.height):
        #                 for indexX in range(boundingBox.width):
        #                     velocities[boundingBox.y + indexY][boundingBox.x + indexX][0] = (velocities[boundingBox.y + indexY][boundingBox.x + indexX][0] + fillingVelocityX) / 2
        #                     velocities[boundingBox.y + indexY][boundingBox.x + indexX][1] = (velocities[boundingBox.y + indexY][boundingBox.x + indexX][1] + fillingVelocityY) / 2

        print('Calculated velocities')
        # index 3 flow matrix x velocity index 4 flow matrix y velocity
        for key, track in tracks.items():
            boundingBoxIndex = 0
            for trackedBoundingBox in track.boundingBoxes:
                if boundingBoxIndex < 1:
                    boundingBoxIndex += 1
                    continue
                if boundingBoxIndex + 1 == len(track.boundingBoxes):
                    break
                array.fill(0)
                # calculate velocity
                def calculateVelocity(leftBox, rightBox):
                    velocityX = (rightBox.x - leftBox.x)\
                            / float(abs(leftBox.frameId - rightBox.frameId))
                    velocityY = (rightBox.y - leftBox.y)\
                            / float(abs(leftBox.frameId - rightBox.frameId))
                    return velocityX, velocityY
                def setPreviousVelocity(leftBox, rightBox, array, middleX, middleY, index):
                    prevVelocityX, prevVelocityY = calculateVelocity(leftBox, rightBox)
                    sameFrameMiddleX = int(rightBox.x + rightBox.width / 2)
                    sameFrameMiddleY = int(rightBox.y + rightBox.height / 2)
                    indexX = sameFrameMiddleX - (middleX - 30)
                    if indexX < 0 or indexX > 59:
                        print('Outside of bounds : ' + str(indexX) + ' VelocityX: ' + str(prevVelocityX) + ' VelocityY: ' + str(prevVelocityY))
                        return
                    indexY = sameFrameMiddleY - (middleY - 30)
                    if indexY < 0 or indexY > 59:
                        print('Outside of bounds : ' + str(indexY) + ' VelocityX: ' + str(prevVelocityX) + ' VelocityY: ' + str(prevVelocityY))
                        return
                    array[indexY][indexX][index] = prevVelocityX
                    array[indexY][indexX][index + 1] = prevVelocityY
                previousBoundingBox = track.boundingBoxes[boundingBoxIndex - 1]
                currentVelocityX, currentVelocityY = calculateVelocity(previousBoundingBox, trackedBoundingBox)
                nextBoundingBox = track.boundingBoxes[boundingBoxIndex + 1]
                futureVelocityX, futureVelocityY = calculateVelocity(trackedBoundingBox, nextBoundingBox)
                array[30][30][0] = currentVelocityX
                array[30][30][1] = currentVelocityY

                # fill other cells
                middleX = int(trackedBoundingBox.x + trackedBoundingBox.width / 2)
                middleY = int(trackedBoundingBox.y + trackedBoundingBox.height / 2)
                # for i in range(60):
                #     for k in range(60):
                #         array[i][k][2] = velocities[middleY - 30 + i][middleX - 30 + k][0]
                #         array[i][k][3] = velocities[middleY - 30 + i][middleX - 30 + k][1]
                # setPreviousVelocity(track.boundingBoxes[boundingBoxIndex - 2],
                #                     track.boundingBoxes[boundingBoxIndex - 1], array, middleX, middleY, 2)
                # setPreviousVelocity(track.boundingBoxes[boundingBoxIndex - 3],
                #                     track.boundingBoxes[boundingBoxIndex - 2], array, middleX, middleY, 4)
                # setPreviousVelocity(track.boundingBoxes[boundingBoxIndex - 4],
                #                     track.boundingBoxes[boundingBoxIndex - 3], array, middleX, middleY, 6)

                # frameDiff = track.boundingBoxes[boundingBoxIndex - 4].frameId - trackedBoundingBox.frameId
                # if frameDiff > 4:
                #     print('Frame diff: ' + str(frameDiff))

                for sameFrameBoundingBox in annotatedData[trackedBoundingBox.frameId].boundingBoxes:
                    sameFrameMiddleX = sameFrameBoundingBox.x + sameFrameBoundingBox.width / 2
                    if middleX - 31 < sameFrameMiddleX < middleX + 30:
                        sameFrameMiddleY = sameFrameBoundingBox.y + sameFrameBoundingBox.height / 2
                        if middleY - 31 < sameFrameMiddleY < middleY + 30:
                            array[int(sameFrameMiddleY - (middleY - 30))][int(sameFrameMiddleX - (middleX - 30))][2] = 1

                # fill flow matrix
                for i in range(60):
                    if 0 <= trackedBoundingBox.x - 30 + i < len(flowmatrix):
                        for k in range(60):
                            if 0 <= trackedBoundingBox.y - 30 + k < len(flowmatrix[0]):
                                data = flowmatrix[trackedBoundingBox.x - 30 + i][trackedBoundingBox.y - 30 + k]
                                if data[0] == -1:
                                    array[k][i][3] = 0
                                    array[k][i][4] = 0
                                else:
                                    array[k][i][3] = data[1][0]
                                    array[k][i][4] = data[1][1]

                tf_example = self.createTrackingTFRecord(array, 60, 60, trackedBoundingBox.x, trackedBoundingBox.y, [futureVelocityX, futureVelocityY])
                writers[0].write(tf_example.SerializeToString())
                boundingBoxIndex += 1
                entries += 1
            print('Processed track: ' + str(key))
        print('Saved ' + str(entries) + ' to dataset.')

    def loadFromDataset(self, filename):
        record = tf.data.TFRecordDataset(filename)
        feature_description = {
            'data/height': tf.FixedLenFeature([], tf.int64),
            'data/width': tf.FixedLenFeature([], tf.int64),
            'data/x': tf.FixedLenFeature([], tf.int64),
            'data/y': tf.FixedLenFeature([], tf.int64),
            'data/features': tf.FixedLenFeature([60, 60, 5], tf.float32),
            'data/response': tf.FixedLenFeature([2], tf.float32),
        }

        def _parse_function(example_proto):
            # Parse the input tf.Example proto using the dictionary above.
            return tf.parse_single_example(example_proto, feature_description)

        heights = []
        widths = []
        xs = []
        ys = []
        featuresList = []
        responsesList = []
        dataset = record.map(_parse_function)
        iterator = dataset.make_one_shot_iterator()
        next_image_data = iterator.get_next()
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            try:
                while True:
                    data = sess.run(next_image_data)
                    heights.append(data['data/height'])
                    widths.append(data['data/width'])
                    xs.append(data['data/x'])
                    ys.append(data['data/y'])
                    featuresList.append(data['data/features'])
                    responsesList.append(data['data/response'])
            except:
                pass
        # for data in dataset:
        #     heights.append(data['data/height'].numpy())
        #     widths.append(data['data/width'].numpy())
        #     featuresList.append(data['data/features'].numpy())
        #     responsesList.append(data['data/response'].numpy())

        features = np.stack(featuresList)
        responses = np.stack(responsesList)

        return TrackingDataset(widths, heights, xs, ys, features, responses)


if __name__ == "__main__":
    dataset = Dataset()
    annotatedData = []
    XMLRead.readXML('C:\\GitHubCode\\phd\\ImageCytometry\\src\\XML\\tracks_1_300.xml', annotatedData)
    # tracks, mat, src_names = XMLRead.parseXMLDataForTracks(annotatedData, True)
    # flowMatrix = CellDataReader.FlowMatrix(1280, 720)
    # unresolved_from_tracking = []
    # flow_matrix = flowMatrix.oldFlowMatrix(tracks, unresolved_from_tracking)
    flowMatrixNew = CellDataReader.FlowMatrix(1280, 720, 3)
    flowMatrixNew.readFlowMatrix(Definitions.DATA_ROOT_DIRECTORY + Definitions.FLOW_MATRIX_FILE)
    flow_matrix = flowMatrixNew.convertToOldArrayType()
    newDir = 'C:\\GitHubCode\\phd\\ImageCytometry\\src\\TFRecord\\tracking'
    if not os.path.exists(newDir):
        os.makedirs(newDir)
    dataset.createTrackingDataset('C:\\GitHubCode\\phd\\ImageCytometry\\src\\TFRecord\\tracking\\', annotatedData, 'Tracking250SimulationMatrixFixed', flow_matrix)
