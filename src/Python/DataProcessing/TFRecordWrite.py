import tensorflow as tf
import XMLRead
import cv2
from object_detection.utils import dataset_util

flags = tf.app.flags
flags.DEFINE_string('output_path', 'C:\\GitHubCode\\phd\\ImageCytometry\\src\\TFRecord\\outputRecord200',
                    'Path to output TFRecord')
FLAGS = flags.FLAGS

datadir = 'D:\\BigData\\cellinfluid\\bunkyObrazkyTiff\\'


def create_tf_example(imageData):
    # image = cv2.imread(datadir + imageData.filename, cv2.IMREAD_GRAYSCALE)
    image = cv2.imread(datadir + imageData.filename, cv2.IMREAD_COLOR)
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


def main(_):
    writer = tf.python_io.TFRecordWriter(FLAGS.output_path + '1' + '.record')

    # TODO(user): Write code to read in your dataset to examples variable
    examples = []
    filePathImageData = datadir + 'tracks_1_200.xml'
    XMLRead.readXML(filePathImageData, examples)
    i = 0
    for example in examples:
        i = i + 1
        tf_example = create_tf_example(example)
        writer.write(tf_example.SerializeToString())
        if i == 150:
            writer.close()
            writer = tf.python_io.TFRecordWriter(FLAGS.output_path + '2' + '.record')
        elif i == 201:
            break

    writer.close()


if __name__ == '__main__':
    tf.app.run()
