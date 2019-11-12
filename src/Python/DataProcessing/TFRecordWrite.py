import tensorflow as tf
import XMLRead
import cv2
from object_detection.utils import dataset_util

FOLDER_PATH = 'C:\\GitHubCode\\phd\\ImageCytometry\\src\\TFRecord\\'
DATA_RECORD_NAME = '250NoBackground'

BACKGROUND = True
imageFolders = []
# imageFolders.append('D:\\BigData\\cellinfluid\\bunkyObrazkyTiff\\')
imageFolders.append('D:\\BigData\\cellinfluid\\bunkyObrazkyTiff\\subtractedBackgrounds\\')
# imageFolders.append('D:\\BigData\\cellinfluid\\deformabilityObrazky\\')
# imageFolders.append('D:\\BigData\\cellinfluid\\deformabilityObrazky\\subtractedBackgrounds\\')
xmlFiles = []
xmlFiles.append('tracks_1_300.xml')
# xmlFiles.append('deformabilityAnnotations.xml')
PATH_TO_ANNOTATED_DATA = 'C:\\GitHubCode\\phd\\ImageCytometry\\src\\XML\\'

stopPoints = []
stopPoints.append((200, 250))
stopPoints.append((30, 51))

def create_tf_example(imageData, imagePath):
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

def processXML(writers, annotatedData, boundary, end, index):
    i = 0
    writerIndex = 0
    for example in annotatedData:
        i = i + 1
        tf_example = create_tf_example(example, imageFolders[index])
        writers[writerIndex].write(tf_example.SerializeToString())
        if i == boundary:
            writerIndex += 1
        elif i == end:
            break

def main(_):
    writers = []
    writers.append(tf.python_io.TFRecordWriter(FOLDER_PATH + 'train' + DATA_RECORD_NAME + '.record'))
    writers.append(tf.python_io.TFRecordWriter(FOLDER_PATH + 'eval' + DATA_RECORD_NAME + '.record'))

    i = 0
    while i < len(imageFolders):
        annotatedData = []
        filePathImageData = PATH_TO_ANNOTATED_DATA + xmlFiles[i]
        XMLRead.readXML(filePathImageData, annotatedData)
        processXML(writers, annotatedData, stopPoints[i][0], stopPoints[i][1], i)
        i += 1

    writers[0].close()
    writers[1].close()

if __name__ == '__main__':
    tf.app.run()
