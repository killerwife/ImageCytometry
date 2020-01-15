import tensorflow as tf
import numpy as np
import os,glob,cv2
import Dataset
import math
import sys

# Step-1: Recreate the network graph. At this step only graph is created.
saver = tf.train.import_meta_graph('./trainingOutput/trackingNeuralNetwork.meta')
# Let us restore the saved model
sess = tf.Session()
# Step-2: Now let's load the weights saved using the restore method.
saver.restore(sess, tf.train.latest_checkpoint('./trainingOutput/'))

dataset = Dataset.Dataset()
trackingDataset = dataset.loadFromDataset('C:\\GitHubCode\\phd\\ImageCytometry\\src\\TFRecord\\tracking\\trainTracking250SimulationMatrixFixed.record')

allSamples = len(trackingDataset.features)
testingStart = int(allSamples * 80 / 100)

inputData = trackingDataset.features[testingStart: testingStart + 32]
outputData = trackingDataset.response[testingStart: testingStart + 32]

# Accessing the default graph which we have restored
graph = tf.get_default_graph()

# Now, let's get hold of the op that we can be processed to get the output.
# In the original network y_pred is the tensor that is the prediction of the network
y_pred = graph.get_tensor_by_name("y_pred:0")

# Let's feed the images to the input placeholders
x = graph.get_tensor_by_name("x:0")
y_true = graph.get_tensor_by_name("y_true:0")
y_test_images = np.zeros((32, 2))


# Creating the feed_dict that is required to be fed to calculate y_pred
feed_dict_testing = {x: inputData, y_true: y_test_images}
result = sess.run(y_pred, feed_dict=feed_dict_testing)
# result is of this format [probabiliy_of_rose probability_of_sunflower]
error = 0
for i in range(len(result)):
    error += abs(result[i][0] - outputData[i][0]) + abs(result[i][1] - outputData[i][1])

np.set_printoptions(threshold=sys.maxsize)
print('Error: ' + str(error))

# for l in range(5):
#     for i in range(len(inputData[0])):
#         for k in range(len(inputData[0][i])):
#             print(inputData[0][i][k][l], end=" ")
#         print()
#     print()

for i in range(len(result)):
    print('Original data: ' + str(outputData[i]) + ' Predicted data: ' + str(result[i]))

