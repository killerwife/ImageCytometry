import numpy as np
import os
import six.moves.urllib as urllib
import sys
import tarfile
import tensorflow as tf
import zipfile
import cv2

from distutils.version import StrictVersion
from collections import defaultdict
from io import StringIO
# This is needed to display the images.
from matplotlib import pyplot as plt
from PIL import Image
from tensorflow.python.saved_model import loader
from tensorflow.python.platform import gfile
from google.protobuf import text_format

# This is needed since the notebook is stored in the object_detection folder.
sys.path.append("..")
from object_detection.utils import ops as utils_ops

if StrictVersion(tf.__version__) < StrictVersion('1.9.0'):
  raise ImportError('Please upgrade your TensorFlow installation to v1.9.* or later!')

from utils import label_map_util

from utils import visualization_utils as vis_util

# List of the strings that is used to add correct label for each box.
PATH_TO_LABELS = os.path.join('data', 'C:\\GitHubCode\\phd\\exportedModels\\label_map.txt')

category_index = label_map_util.create_category_index_from_labelmap(PATH_TO_LABELS, use_display_name=True)

def load_image_into_numpy_array(image):
  (im_width, im_height) = image.size
  return np.array(image.getdata()).reshape(
      (im_height, im_width, 3)).astype(np.uint8)

# For the sake of simplicity we will use only 2 images:
# image1.jpg
# image2.jpg
# If you want to test the code with your images, just add path to the images to the TEST_IMAGE_PATHS.
# PATH_TO_TEST_IMAGES_DIR = 'D:\\BigData\\cellinfluid\\bunkyObrazkyTiff\\501-2992'
# TEST_IMAGE_PATHS = [ os.path.join(PATH_TO_TEST_IMAGES_DIR, 'video2359_050{}.tiff'.format(i)) for i in range(1, 5) ]
# IMAGE_PREFIX = 'tiff'
PATH_TO_TEST_IMAGES_DIR = 'D:\\BigData\\cellinfluid\\deformabilityObrazky\\1-50'
TEST_IMAGE_PATHS = [ os.path.join(PATH_TO_TEST_IMAGES_DIR, 'image{}.png'.format(i)) for i in range(0, 5) ]
IMAGE_PREFIX = 'deformability'
# Size, in inches, of the output images.
IMAGE_SIZE = (12, 8)

MODEL_NAME = 'model21032019_02-200Gray'
GRAYSCALE = True
# Path to frozen detection graph. This is the actual model that is used for the object detection.
PATH_TO_FROZEN_GRAPH = 'C:\\GitHubCode\\phd\\exportedModels\\' + MODEL_NAME + '\\frozen_inference_graph.pb'

detection_graph = tf.Graph()
with detection_graph.as_default():
  od_graph_def = tf.GraphDef()
  with tf.gfile.GFile(PATH_TO_FROZEN_GRAPH, 'rb') as fid:
    serialized_graph = fid.read()
    od_graph_def.ParseFromString(serialized_graph)
    tf.import_graph_def(od_graph_def, name='')


def run_inference_for_array(imageArray, graph):
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
            output_dict = sess.run(tensor_dict,
                                 feed_dict={image_tensor: imageArray})

            # all outputs are float32 numpy arrays, so convert types as appropriate
            # output_dict['num_detections'] = int(output_dict['num_detections'][0])
            # output_dict['detection_classes'] = output_dict[
            #     'detection_classes'][0].astype(np.uint8)
            # output_dict['detection_boxes'] = output_dict['detection_boxes'][0]
            # output_dict['detection_scores'] = output_dict['detection_scores'][0]
            # if 'detection_masks' in output_dict:
            #   output_dict['detection_masks'] = output_dict['detection_masks'][0]
    return output_dict


if not os.path.exists(MODEL_NAME):
    os.makedirs(MODEL_NAME)

i = 0
imageArray = []
for image_path in TEST_IMAGE_PATHS:
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
    imageArray.append(image_np)

output_dict = run_inference_for_array(imageArray, detection_graph)
for i in range(0, len(imageArray)):
    # Visualization of the results of a detection.
    vis_util.visualize_boxes_and_labels_on_image_array(
        imageArray[i],
        output_dict['detection_boxes'][i],
        output_dict['detection_classes'][i].astype(np.uint8),
        output_dict['detection_scores'][i],
        category_index,
        instance_masks=output_dict.get('detection_masks'),
        use_normalized_coordinates=True,
        line_thickness=8,
        max_boxes_to_draw=None)
    plt.figure(figsize=IMAGE_SIZE)
    # cv2.imshow('image', image_np)
    cv2.imwrite(MODEL_NAME + '\\' + IMAGE_PREFIX + str(i) + '.png', imageArray[i])
    i = i + 1

