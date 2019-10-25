import numpy as np
import os
import sys
import tensorflow as tf
import cv2
import XMLRead
from distutils.version import StrictVersion
from object_detection.utils import ops as utils_ops
from utils import label_map_util
from PIL import Image

sys.path.append("..")

if StrictVersion(tf.__version__) < StrictVersion('1.9.0'):
    raise ImportError('Please upgrade your TensorFlow installation to v1.9.* or later!')

from utils import visualization_utils as vis_util

PATH_TO_LABELS = os.path.join('data', 'C:\\GitHubCode\\phd\\exportedModels\\label_map.txt')

category_index = label_map_util.create_category_index_from_labelmap(PATH_TO_LABELS, use_display_name=True)


def load_image_into_numpy_array(image):
    (im_width, im_height) = image.size
    return np.array(image.getdata()).reshape(
        (im_height, im_width, 3)).astype(np.uint8)


# PATH_TO_TEST_IMAGES_DIR = 'D:\\BigData\\cellinfluid\\bunkyObrazkyTiff\\501-2992'
# TEST_IMAGE_PATHS = [ os.path.join(PATH_TO_TEST_IMAGES_DIR, 'video2359_050{}.tiff'.format(i)) for i in range(1, 10) ]
# IMAGE_PREFIX = 'tiff'
# PATH_TO_TEST_IMAGES_DIR = 'D:\\BigData\\cellinfluid\\deformabilityObrazky'
# TEST_IMAGE_PATHS = [ os.path.join(PATH_TO_TEST_IMAGES_DIR, 'image{}.png'.format(i)) for i in range(0, 10) ]
# IMAGE_PREFIX = 'deformability'
# Size, in inches, of the output images.
PATH_TO_ANNOTATIONS = 'C:\\GitHubCode\\phd\\ImageCytometry\\src\\XML\\'
PATH_TO_PREDICTED_ANNOTATIONS = 'D:\\BigData\\cellinfluid\\Annotations\\PredictedAnnotations\\'

MODEL_NAME = 'model08042019-250And50'
GRAYSCALE = False
BACKGROUND = False
FIRST_VIDEO = False
PATH_TO_FROZEN_GRAPH = 'C:\\GitHubCode\\phd\\exportedModels\\' + MODEL_NAME + '\\frozen_inference_graph.pb'
if FIRST_VIDEO == True:
    ANNOTATIONS_FILE_NAME = 'tracks_1_300.xml'
    if BACKGROUND == False:
        PATH_TO_IMAGE_ROOT_DIR = 'D:\\BigData\\cellinfluid\\bunkyObrazkyTiff\\'  # regular
    else:
        PATH_TO_IMAGE_ROOT_DIR = 'D:\\BigData\\cellinfluid\\subtractedBackgrounds\\' # no background
else:
    ANNOTATIONS_FILE_NAME = 'deformabilityAnnotations.xml'
    if BACKGROUND == False:
        PATH_TO_IMAGE_ROOT_DIR = 'D:\\BigData\\cellinfluid\\\deformabilityObrazky\\'  # regular
    else:
        PATH_TO_IMAGE_ROOT_DIR = 'D:\\BigData\\cellinfluid\\\deformabilityObrazky\\subtractedBackgrounds\\' # no background - TODO
    MODEL_NAME += 'SecondVideo'

IMAGE_BATCH = 5

detection_graph = tf.Graph()
with detection_graph.as_default():
    od_graph_def = tf.GraphDef()
    with tf.gfile.GFile(PATH_TO_FROZEN_GRAPH, 'rb') as fid:
        serialized_graph = fid.read()
        od_graph_def.ParseFromString(serialized_graph)
        tf.import_graph_def(od_graph_def, name='')


def run_inference_for_all_files(graph, fileNames):
    outputData = []
    with graph.as_default():
        with tf.Session() as sess:
            # Get handles to input and output tensors
            ops = tf.get_default_graph().get_operations()
            all_tensor_names = {output.name for op in ops for output in op.outputs}
            tensor_dict = {}
            for key in [
                'num_detections', 'detection_boxes', 'detection_scores',
                'detection_classes', 'detection_masks'
            ]:
                tensor_name = key + ':0'
                if tensor_name in all_tensor_names:
                    tensor_dict[key] = tf.get_default_graph().get_tensor_by_name(
                        tensor_name)
            if 'detection_masks' in tensor_dict:
                # The following processing is only for single image
                detection_boxes = tf.squeeze(tensor_dict['detection_boxes'], [0])
                detection_masks = tf.squeeze(tensor_dict['detection_masks'], [0])
                # Reframe is required to translate mask from box coordinates to image coordinates and fit the image size.
                real_num_detection = tf.cast(tensor_dict['num_detections'][0], tf.int32)
                detection_boxes = tf.slice(detection_boxes, [0, 0], [real_num_detection, -1])
                detection_masks = tf.slice(detection_masks, [0, 0, 0], [real_num_detection, -1, -1])
                detection_masks_reframed = utils_ops.reframe_box_masks_to_image_masks(
                    detection_masks, detection_boxes, imageArray[0].shape[0], imageArray[0].shape[1])
                detection_masks_reframed = tf.cast(
                    tf.greater(detection_masks_reframed, 0.5), tf.uint8)
                # Follow the convention by adding back the batch dimension
                tensor_dict['detection_masks'] = tf.expand_dims(
                    detection_masks_reframed, 0)
            image_tensor = tf.get_default_graph().get_tensor_by_name('image_tensor:0')

            # Run inference
            imageIndex = 0
            imageNameIndex = 0
            for image_path in fileNames:
                if GRAYSCALE:
                    # greyscale read
                    image_np = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
                    image_np = np.expand_dims(image_np, -1)
                    dim_to_repeat = 2
                    repeats = 3
                    image_np = np.repeat(image_np, repeats, dim_to_repeat)
                else:
                    # color read
                    image = Image.open(image_path)
                    image_np = load_image_into_numpy_array(image)
                # Actual detection.
                imageArray.append(image_np)
                if len(imageArray) == IMAGE_BATCH:
                    imageBatchIndex = 0
                    output_dict = sess.run(tensor_dict, feed_dict={image_tensor: imageArray})
                    im_width = image_np.shape[1]
                    im_height = image_np.shape[0]
                    while imageBatchIndex < IMAGE_BATCH:
                        imageData = XMLRead.Image()
                        imageData.filename = fileNames[imageNameIndex]
                        boxIndex = 0
                        arraySize = len(output_dict['detection_boxes'][imageBatchIndex])
                        while boxIndex < arraySize:
                            if output_dict['detection_scores'][imageBatchIndex][boxIndex] > 0.5:
                                boundingBoxTensor = output_dict['detection_boxes'][imageBatchIndex][boxIndex]
                                boundingBox = XMLRead.BoundingBox()
                                boundingBox.x = int(boundingBoxTensor[1] * im_width)
                                boundingBox.y = int(boundingBoxTensor[0] * im_height)
                                boundingBox.width = int(boundingBoxTensor[3] * im_width - boundingBox.x)
                                boundingBox.height = int(boundingBoxTensor[2] * im_height - boundingBox.y)
                                boundingBox.trackId = -1
                                imageData.boundingBoxes.append(boundingBox)
                            boxIndex = boxIndex + 1
                        imageBatchIndex = imageBatchIndex + 1
                        imageNameIndex = imageNameIndex + 1
                        outputData.append(imageData)
                    imageArray.clear()
                imageIndex = imageIndex + 1
                # if imageIndex > 20:
                #     break
    return outputData


i = 0
imageArray = []
fileNamePredicted = PATH_TO_ANNOTATIONS + ANNOTATIONS_FILE_NAME
fileNameOutput = PATH_TO_PREDICTED_ANNOTATIONS + 'tracks_1_300_' + MODEL_NAME + '.xml'
annotatedData = []
XMLRead.readXML(fileNamePredicted, annotatedData)
fileNames = []
for data in annotatedData:
    fileNames.append(PATH_TO_IMAGE_ROOT_DIR + data.filename)

outputData = run_inference_for_all_files(detection_graph, fileNames)
XMLRead.writeXML(outputData, fileNameOutput)