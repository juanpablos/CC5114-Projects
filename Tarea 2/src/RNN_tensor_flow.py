import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf

matplotlib.style.use('ggplot')


def get_values(dictionary, key_arr):
    w = []
    for key in key_arr:
        w.append(dictionary[key])
    return w


def collect(a_list):
    return [[a_list[ii][j] for ii in range(len(a_list))] for j in range(len(a_list[0]))]


def make_plots(data1, data2, epoch, sequence, layer_size, _network_type, repo, to_predict):
    options_titles = {
        'new_files_pred': 'Predictions for new files.',
        'deleted_files_pred': 'Predictions for deleted files.',
        'modified_files_pred': 'Predictions for modified files.',
        'inserted_lines_pred': 'Predictions for inserted lines.',
        'deleted_lines_pred': 'Predictions for deleted lines.'
    }
    for d1, d2, t, index in zip(data1.columns, data2.columns, get_values(options_titles, to_predict),
                                range(len(to_predict))):
        ax = data1.plot(kind='line', x='x', y=d1)
        data2.plot(kind='line', x='x', y=d2, ax=ax)
        plt.title("{} repo. {} {} with layer size {}, sequence {}, {} epochs".format(repo, t, _network_type, layer_size,
                                                                          sequence, epoch))
        figure = plt.gcf()
        figure.set_size_inches(19, 10)
        plt.savefig('Images/{}-{}-{}'.format(repo, _network_type, to_predict[index]), bbox_inches='tight', dpi=200)
        plt.close()


def make_loss_plot(loss_data, repo, _network_type, layer_size, sequence, epochs, to_predict):
    plt.title("{} repo. Loss of {} with layer size {}, sequence {}, {} epochs".format(repo, _network_type,
                                                                           layer_size,
                                                                           sequence, epochs))
    plt.scatter(x=np.arange(0, len(loss_data)), y=loss_data)
    plt.xlabel('epochs')
    plt.ylabel('loss')
    figure = plt.gcf()
    figure.set_size_inches(19, 10)
    plt.savefig('Images/{}-{}-loss-{}'.format(repo, _network_type, to_predict), bbox_inches='tight', dpi=200)
    plt.close()


def run_the_model(repository, network_type, to_predict, epoch=1, sequence=3):
    if repository == "osu":
        data_file = 'osu_out.csv'
    elif repository == "cython":
        data_file = 'cython_out.csv'
    elif repository == "tensor flow":
        data_file = 'tensor_out.csv'
    elif repository == "vlc":
        data_file = 'vlc_out.csv'
    elif repository == "php":
        data_file = 'php_out.csv'
    elif repository == "scala":
        data_file = 'scala_out.csv'
    elif repository == "scala":
        data_file = 'scala_out.csv'
    elif repository == "swift":
        data_file = 'swift_out.csv'
    else:
        data_file = None

    dataset = pd.read_csv(data_file)
    datasetNorm = (dataset - dataset.mean()) / (dataset.max() - dataset.min())

    num_epochs = epoch
    batch_size = 1
    total_series_length = len(dataset.index)
    sequence_length = sequence  # The size of the sequence
    neurons_layer = 20  # The number of neurons
    num_features = 5  # input neurons
    num_classes = len(to_predict)  # output neurons

    min_test_size = 0.05

    datasetTrain = datasetNorm[dataset.index < (total_series_length * (1 - min_test_size))]
    test_first_idx = total_series_length * (1 - min_test_size)
    datasetTest = datasetNorm[dataset.index >= test_first_idx]

    X_train = datasetTrain[['new_files', 'deleted_files', 'modified_files', 'inserted_lines', 'deleted_lines']]
    X_test = datasetTest[['new_files', 'deleted_files', 'modified_files', 'inserted_lines', 'deleted_lines']]

    y_train = datasetTrain[to_predict]
    y_test = datasetTest[to_predict]

    xTrain = X_train.as_matrix()
    yTrain = y_train.as_matrix()

    xTest = X_test.as_matrix()
    yTest = y_test.as_matrix()

    num_batches = len(xTrain) // sequence_length

    print('The total series length is: %d' % total_series_length)
    print('The current configuration gives us %d batches looking %d steps in the past'
          % (num_batches, sequence_length))
    print('Tests have {} batches'.format(len(xTest) // sequence_length))

    batchX_placeholder = tf.placeholder(dtype=tf.float32, shape=[None, sequence_length, num_features],
                                        name='data_ph')
    batchY_placeholder = tf.placeholder(dtype=tf.float32, shape=[None, sequence_length, num_classes],
                                        name='target_ph')

    labels_series = tf.unstack(batchY_placeholder, axis=1)

    if network_type == "RNN":
        cell = tf.contrib.rnn.BasicRNNCell(num_units=neurons_layer)
    elif network_type == "LSTM":
        cell = tf.contrib.rnn.BasicLSTMCell(num_units=neurons_layer)
    else:
        cell = None

    states_series, current_state = tf.nn.dynamic_rnn(cell=cell, inputs=batchX_placeholder, dtype=tf.float32)
    states_series = tf.transpose(states_series, [1, 0, 2])

    last_state = tf.gather(params=states_series, indices=states_series.get_shape()[0] - 1)
    last_label = tf.gather(params=labels_series, indices=len(labels_series) - 1)

    weight = tf.Variable(tf.truncated_normal([neurons_layer, num_classes]))
    bias = tf.Variable(tf.constant(0.1, shape=[num_classes]))

    prediction = tf.matmul(last_state, weight) + bias

    loss = tf.reduce_mean(tf.squared_difference(last_label, prediction))

    train_step = tf.train.AdamOptimizer(learning_rate=0.01).minimize(loss)

    loss_list = []
    test_pred_list = []

    with tf.Session() as sess:
        tf.global_variables_initializer().run()

        for epoch_idx in range(num_epochs):

            print('Epoch %d' % epoch_idx)

            for batch_idx in range(num_batches):
                start_idx = batch_idx * sequence_length
                end_idx = start_idx + sequence_length * batch_size

                try:
                    batchX = xTrain[start_idx:end_idx, :].reshape(batch_size, sequence_length, num_features)
                    batchY = yTrain[start_idx:end_idx].reshape(batch_size, sequence_length, num_classes)
                except:
                    continue

                feed = {batchX_placeholder: batchX, batchY_placeholder: batchY}

                # TRAIN!
                _loss, _train_step, _pred, _last_label, _prediction = sess.run(
                    fetches=[loss, train_step, prediction, last_label, prediction],
                    feed_dict=feed
                )

                loss_list.append(_loss)

                if batch_idx % 1000 == 0:
                    print('Step %d - Loss: %.6f' % (batch_idx, _loss))

        for test_idx in range(len(xTest) - sequence_length):
            testBatchX = xTest[test_idx:test_idx + sequence_length, :].reshape(
                (1, sequence_length, num_features))
            testBatchY = yTest[test_idx:test_idx + sequence_length].reshape((1, sequence_length, num_classes))
            feed = {batchX_placeholder: testBatchX,
                    batchY_placeholder: testBatchY}
            _last_state, _last_label, test_pred = sess.run([last_state, last_label, prediction], feed_dict=feed)
            test_pred_list.append(test_pred[0])

    make_loss_plot(loss_list, repository, network_type, neurons_layer, sequence_length, num_epochs, to_predict)

    options_prediction = {
        'new_files_pred': 'new_files_prediction',
        'deleted_files_pred': 'deleted_files_prediction',
        'modified_files_pred': 'modified_files_prediction',
        'inserted_lines_pred': 'inserted_line_prediction',
        'deleted_lines_pred': 'deleted_lines_prediction'
    }

    predicted = pd.DataFrame(test_pred_list, columns=get_values(options_prediction, to_predict))

    options_expected = {
        'new_files_pred': 'new_files_expected',
        'deleted_files_pred': 'deleted_files_expected',
        'modified_files_pred': 'modified_files_expected',
        'inserted_lines_pred': 'inserted_line_expected',
        'deleted_lines_pred': 'deleted_lines_expected'
    }

    y_test.columns = get_values(options_expected, to_predict)

    y_test['x'] = list(range(1, len(y_test.index) + 1))
    predicted['x'] = list(range(1, len(predicted.index) + 1))

    make_plots(y_test, predicted, num_epochs, sequence_length, neurons_layer, network_type, repository, to_predict)

    cell = None


if __name__ == '__main__':  # im too lazy to run these myself

    epoch = 50
    seq_length = 3
    repos_to_do = [1, 2, 3, 4, 5, 6, 7]
    networks_use = [1, 2]
    predictions_to_do = [3,4,5]

    repos = {
        1: 'osu',
        2: 'cython',
        3: 'tensor flow',
        4: 'vlc',
        5: 'php',
        6: 'scala',
        7: 'swift'
    }

    networks = {
        1: 'RNN',
        2: 'LSTM'
    }

    predict = {
        1: 'new_files_pred',
        2: 'deleted_files_pred',
        3: 'modified_files_pred',
        4: 'inserted_lines_pred',
        5: 'deleted_lines_pred'
    }

    for r in get_values(repos, repos_to_do):
        for n in get_values(networks, networks_use):
            run_the_model(r, n, get_values(predict, predictions_to_do), epoch=epoch, sequence = seq_length)
            tf.reset_default_graph()
