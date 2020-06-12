import tensorflow as tf
import Dataset
import random
import os

PATH_TO_DATASETS = 'C:\\GitHubCode\\phd\\ImageCytometry\\src\\TFRecord\\tracking\\'
DATASET_NAME = 'trainTracking250SimulationMatrix31AnnotatedFixedRots.record'
NEURAL_NETWORK_OUTPUT_DIR = 'trainingOutput\\'
if not os.path.exists(NEURAL_NETWORK_OUTPUT_DIR):
    os.makedirs(NEURAL_NETWORK_OUTPUT_DIR)
MODEL_NAME = 'trackingNeuralNetworkAnnotated'
NUM_OF_ITERATIONS = 1000000
img_size = 31
num_channels = 5
outputSize = 2
batch_size = 32
resume = True


def create_weights(shape):
    return tf.Variable(tf.truncated_normal(shape, stddev=0.05))


def create_biases(size):
    return tf.Variable(tf.constant(0.05, shape=[size]))


def create_convolutional_layer(input, num_input_channels, conv_filter_size, num_filters):
    # We shall define the weights that will be trained using create_weights function.
    weights = create_weights(shape=[conv_filter_size, conv_filter_size, num_input_channels, num_filters])
    # We create biases using the create_biases function. These are also trained.
    biases = create_biases(num_filters)

    # Creating the convolutional layer
    layer = tf.nn.conv2d(input=input, filter=weights, strides=[1, 1, 1, 1], padding='SAME')

    layer += biases

    layer = tf.nn.relu(layer)

    return layer


def create_pooling_layer(input):
    # We shall be using max-pooling.
    layer = tf.nn.max_pool(value=input, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
    # Output of pooling is fed to Relu which is the activation function for us.
    layer = tf.nn.relu(layer)

    return layer


def create_flatten_layer(layer):
    # We know that the shape of the layer will be [batch_size img_size img_size num_channels]
    # But let's get it from the previous layer.
    layer_shape = layer.get_shape()

    # Number of features will be img_height * img_width* num_channels. But we shall calculate it in place of hard-coding it.
    num_features = layer_shape[1:4].num_elements()

    # Now, we Flatten the layer so we shall have to reshape to num_features
    layer = tf.reshape(layer, [-1, num_features])

    return layer


def create_fc_layer(input, num_inputs, num_outputs, use_relu=True):
    # Let's define trainable weights and biases.
    weights = create_weights(shape=[num_inputs, num_outputs])
    biases = create_biases(num_outputs)

    # Fully connected layer takes input x and produces wx+b.Since, these are matrices, we use matmul function in Tensorflow
    layer = tf.matmul(input, weights) + biases
    if use_relu:
        layer = tf.nn.relu(layer)

    return layer


class CellTracker(object):
    # def load(self, filename):
        # TODO

    # def save(self, filename):
        # TODO

    def train(self, dataset):
        print()


def designNeuralNetwork(img_size, num_channels, output_size):
    x = tf.placeholder(tf.float32, shape=[None, img_size, img_size, num_channels], name='x')

    y_true = tf.placeholder(tf.float32, shape=[None, output_size], name='y_true')

    filter_size_conv1 = 3
    num_filters_conv1 = 32

    filter_size_conv2 = 3
    num_filters_conv2 = 32

    filter_size_conv3 = 3
    num_filters_conv3 = 32

    filter_size_conv4 = 3
    num_filters_conv4 = 32

    filter_size_conv5 = 3
    num_filters_conv5 = 32

    filter_size_conv6 = 3
    num_filters_conv6 = 32

    layer_conv1 = create_convolutional_layer(input=x,
                                             num_input_channels=num_channels,
                                             conv_filter_size=filter_size_conv1,
                                             num_filters=num_filters_conv1)
    layer_conv2 = create_convolutional_layer(input=layer_conv1,
                                             num_input_channels=num_filters_conv1,
                                             conv_filter_size=filter_size_conv2,
                                             num_filters=num_filters_conv2)

    layer_conv3 = create_convolutional_layer(input=layer_conv2,
                                             num_input_channels=num_filters_conv2,
                                             conv_filter_size=filter_size_conv3,
                                             num_filters=num_filters_conv3)

    layer_conv4 = create_convolutional_layer(input=layer_conv3,
                                             num_input_channels=num_filters_conv3,
                                             conv_filter_size=filter_size_conv4,
                                             num_filters=num_filters_conv4)

    layer_conv5 = create_convolutional_layer(input=layer_conv4,
                                             num_input_channels=num_filters_conv4,
                                             conv_filter_size=filter_size_conv5,
                                             num_filters=num_filters_conv5)

    layer_conv6 = create_convolutional_layer(input=layer_conv5,
                                             num_input_channels=num_filters_conv5,
                                             conv_filter_size=filter_size_conv6,
                                             num_filters=num_filters_conv6)

    layer_conv7 = create_convolutional_layer(input=layer_conv6,
                                             num_input_channels=num_filters_conv5,
                                             conv_filter_size=filter_size_conv6,
                                             num_filters=num_filters_conv6)

    layer_conv8 = create_convolutional_layer(input=layer_conv7,
                                             num_input_channels=num_filters_conv5,
                                             conv_filter_size=filter_size_conv6,
                                             num_filters=num_filters_conv6)

    layer_conv9 = create_convolutional_layer(input=layer_conv8,
                                             num_input_channels=num_filters_conv5,
                                             conv_filter_size=filter_size_conv6,
                                             num_filters=num_filters_conv6)

    layer_conv10 = create_convolutional_layer(input=layer_conv9,
                                              num_input_channels=num_filters_conv5,
                                              conv_filter_size=filter_size_conv6,
                                              num_filters=num_filters_conv6)

    layer_conv11 = create_convolutional_layer(input=layer_conv10,
                                              num_input_channels=num_filters_conv5,
                                              conv_filter_size=filter_size_conv6,
                                              num_filters=num_filters_conv6)

    layer_conv12 = create_convolutional_layer(input=layer_conv11,
                                              num_input_channels=num_filters_conv5,
                                              conv_filter_size=filter_size_conv6,
                                              num_filters=num_filters_conv6)

    layer_pool1 = create_pooling_layer(input=layer_conv12)

    layer_flat = create_flatten_layer(layer_pool1)

    layer_fc1 = create_fc_layer(input=layer_flat,
                                num_inputs=layer_flat.get_shape()[1:4].num_elements(),
                                num_outputs=outputSize,
                                use_relu=False)

    named_last_layer = tf.identity(layer_fc1, name="y_pred")

    cost = tf.reduce_mean(tf.square(tf.subtract(named_last_layer, y_true)), name="cost")
    optimizer = tf.train.AdamOptimizer(learning_rate=1e-4).minimize(cost, name="optimizer")
    accuracy = tf.reduce_sum(tf.abs(tf.subtract(named_last_layer, y_true)), name="accuracy")
    return optimizer, cost, accuracy, x, y_true


def loadNeuralNetwork(modelname):
    saver = tf.train.import_meta_graph(modelname + '.meta')
    graph = tf.get_default_graph()
    optimizer = graph.get_tensor_by_name("cost:0")
    cost = graph.get_tensor_by_name("optimizer:0")
    accuracy = graph.get_tensor_by_name("accuracy:0")
    x = graph.get_tensor_by_name("x:0")
    y_true = graph.get_tensor_by_name("y_true:0")
    return optimizer, cost, accuracy, x, y_true


def show_progress(accuracy, epoch, train_loss, feed_dict_train, feed_dict_validate, val_loss):
    acc = session.run(accuracy, feed_dict=feed_dict_train)
    val_acc = session.run(accuracy, feed_dict=feed_dict_validate)
    msg = "Training Epoch {0} --- Training Loss: {4:.3f} Training Error Sum: {1:6.1f}, Validation Error Sum: {2:6.1f},  Validation Loss: {3:.3f}"
    print(msg.format(epoch + 1, acc, val_acc, val_loss, train_loss))


def train(num_iteration, currentIterations, trackingDataset, batch_size, optimizer, cost, accuracy, x, y_true):
    allSamples = len(trackingDataset.features)
    tf.random.set_random_seed(2)
    validationStart = int(allSamples * 60 / 100)
    validationStop = int(allSamples * 80 / 100)
    print('Started composing dataset')
    x_batch_placeholder = tf.placeholder(tf.float32)
    y_true_placeholder = tf.placeholder(tf.float32)
    tfDatasetFull = tf.data.Dataset.from_tensor_slices((x_batch_placeholder, y_true_placeholder))

    #{'x_batch': trackingDataset.features[:validationStop], 'y_true': trackingDataset.response[:validationStop]}
    print('Started shuffling dataset')
    tfDatasetFull = tfDatasetFull.shuffle(validationStop)
    tfDatasetTraining = tfDatasetFull.take(validationStart)
    tfDatasetValidation = tfDatasetFull.skip(validationStart)

    print('Composed dataset and repeating')
    datasetBatchTrain = tfDatasetTraining.batch(batch_size).repeat().shuffle(int(validationStart / 2))
    iteratorTrain = datasetBatchTrain.make_initializable_iterator()
    datasetBatchValidate = tfDatasetValidation.batch(batch_size).repeat()
    iteratorValidate = datasetBatchValidate.make_initializable_iterator()
    next_elementTrain = iteratorTrain.get_next()
    next_elementValidate = iteratorValidate.get_next()
    print('Dataset done')

    session.run(iteratorTrain.initializer, feed_dict={x_batch_placeholder: trackingDataset.features[:validationStop],
                                                      y_true_placeholder: trackingDataset.response[:validationStop]})

    session.run(iteratorValidate.initializer, feed_dict={x_batch_placeholder: trackingDataset.features[:validationStop],
                                                         y_true_placeholder: trackingDataset.response[:validationStop]})

    for i in range(currentIterations, num_iteration):
        # x_batch = trackingDataset.features[validation:]
        # y_true_batch = trackingDataset.response[validation:]
        # x_valid_batch = trackingDataset.features[:validation]
        # y_valid_batch = trackingDataset.response[:validation]

        resultDatasetTrain_x_batch, resultDatasetTrain_y_true = session.run(next_elementTrain)
        resultDatasetValidate_x_batch, resultDatasetValidate_y_true = session.run(next_elementValidate)

        feed_dict_tr = {x: resultDatasetTrain_x_batch, y_true: resultDatasetTrain_y_true}
        feed_dict_val = {x: resultDatasetValidate_x_batch, y_true: resultDatasetValidate_y_true}

        session.run(optimizer, feed_dict=feed_dict_tr)
        if i % (validationStart) == 0:
            train_loss = session.run(cost, feed_dict=feed_dict_tr)
            val_loss = session.run(cost, feed_dict=feed_dict_val)
            epoch = int(i / int(validationStart))

            show_progress(accuracy, epoch, train_loss, feed_dict_tr, feed_dict_val, val_loss)
            saver.save(session, outputName)
            f = open(outputName + '.txt', "w")
            f.write(str(i))
            f.close()

    currentIterations += num_iteration

session = tf.Session()

dataset = Dataset.Dataset()
trackingDataset = dataset.loadFromDataset(PATH_TO_DATASETS + DATASET_NAME, img_size, num_channels)
outputName = NEURAL_NETWORK_OUTPUT_DIR + MODEL_NAME

optimizer, cost, accuracy, x, y_true = designNeuralNetwork(img_size, num_channels, outputSize)
session.run(tf.global_variables_initializer())

saver = tf.train.Saver()
currentIterations = 1

if resume == True:
    saver.restore(session, tf.train.latest_checkpoint(NEURAL_NETWORK_OUTPUT_DIR))
    with open(outputName + '.txt') as file:
        for line in file:
            lineSplit = line.split()
            currentIterations = int(lineSplit[0])
            currentIterations += 1
# optimizer, cost, accuracy, x, y_true = loadNeuralNetwork(outputName)

train(num_iteration=NUM_OF_ITERATIONS, currentIterations=currentIterations, trackingDataset=trackingDataset, batch_size=batch_size, optimizer=optimizer,
      cost=cost, accuracy=accuracy, x=x, y_true=y_true)

saver.save(session, outputName)
f = open(outputName + '.txt', "w")
f.write(str(NUM_OF_ITERATIONS))
f.close()







