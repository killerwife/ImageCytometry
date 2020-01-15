import tensorflow as tf
import Dataset

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


img_size = 60
num_channels = 9
outputSize = 2

session = tf.Session()

dataset = Dataset.Dataset()
trackingDataset = dataset.loadFromDataset('C:\\GitHubCode\\phd\\ImageCytometry\\src\\TFRecord\\tracking\\trainTracking2504Channels.record')
outputName = '.\\trainingOutput\\trackingNeuralNetwork'

x = tf.placeholder(tf.float32, shape=[None, img_size, img_size, num_channels], name='x')

y_true = tf.placeholder(tf.float32, shape=[None, outputSize], name='y_true')

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

cost = tf.reduce_mean(tf.square(tf.subtract(named_last_layer, y_true)))
optimizer = tf.train.AdamOptimizer(learning_rate=1e-4).minimize(cost)
accuracy = tf.reduce_sum(tf.abs(tf.subtract(named_last_layer, y_true)))

session.run(tf.global_variables_initializer())


def show_progress(epoch, feed_dict_train, feed_dict_validate, val_loss):
    acc = session.run(accuracy, feed_dict=feed_dict_train)
    val_acc = session.run(accuracy, feed_dict=feed_dict_validate)
    msg = "Training Epoch {0} --- Training Accuracy: {1:>6.1%}, Validation Accuracy: {2:>6.1%},  Validation Loss: {3:.3f}"
    print(msg.format(epoch + 1, acc, val_acc, val_loss))


total_iterations = 0

saver = tf.train.Saver()

batch_size = 32

def train(num_iteration, trackingDataset):
    global total_iterations
    allSamples = len(trackingDataset.features)
    validationStart = int(allSamples * 60 / 100)
    validationStop = int(allSamples * 80 / 100)
    tfDatasetTraining = tf.data.Dataset.from_tensor_slices({'x_batch': trackingDataset.features[:validationStart],
                                                            'y_true': trackingDataset.response[:validationStart]})
    tfDatasetValidation = tf.data.Dataset.from_tensor_slices({'x_batch': trackingDataset.features[validationStart:validationStop],
                                                              'y_true': trackingDataset.response[validationStart:validationStop]})

    datasetBatchTrain = tfDatasetTraining.batch(batch_size).repeat().shuffle(int(validationStart / 2))
    iteratorTrain = datasetBatchTrain.make_one_shot_iterator()
    datasetBatchValidate = tfDatasetValidation.batch(batch_size).repeat()
    iteratorValidate = datasetBatchValidate.make_one_shot_iterator()
    next_elementTrain = iteratorTrain.get_next()
    next_elementValidate = iteratorValidate.get_next()

    for i in range(total_iterations, total_iterations + num_iteration):
        # x_batch = trackingDataset.features[validation:]
        # y_true_batch = trackingDataset.response[validation:]
        # x_valid_batch = trackingDataset.features[:validation]
        # y_valid_batch = trackingDataset.response[:validation]

        resultDatasetTrain = session.run(next_elementTrain)
        resultDatasetValidate = session.run(next_elementValidate)

        feed_dict_tr = {x: resultDatasetTrain['x_batch'], y_true: resultDatasetTrain['y_true']}
        feed_dict_val = {x: resultDatasetValidate['x_batch'], y_true: resultDatasetValidate['y_true']}

        session.run(optimizer, feed_dict=feed_dict_tr)
        if i % 400 == 0:
            val_loss = session.run(cost, feed_dict=feed_dict_val)
            epoch = int(i / int(validationStart / batch_size))

            show_progress(epoch, feed_dict_tr, feed_dict_val, val_loss)
            saver.save(session, outputName)

    total_iterations += num_iteration

train(num_iteration=10000, trackingDataset=trackingDataset)







