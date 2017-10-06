import tensorflow as tf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.style.use('ggplot')


def collect(a_list):
    return [[a_list[i][j] for i in range(len(a_list))] for j in range(len(a_list[0]))]


def make_plots(data1, data2, epoch, sequence, layer_size, network_type, repo):
    titles = ['Predictions for new files.', 'Predictions for deleted files.', 'Predictions for modified files.',
              'Predictions for inserted lines.', 'Predictions for deleted lines.']
    for d1, d2, t in zip(data1.columns, data2.columns, titles):
        ax = data1.plot(kind='line', x='x', y=d1)
        data2.plot(kind='line', x='x', y=d2, ax=ax)
        plt.title("{} repo. {} {} with layer size {}, sequence {}, {} epochs".format(repo, t, network_type, layer_size, sequence, epoch))
        plt.show()


repository = "vlc"

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
elif repository == "swift":
    data_file = 'swift_out.csv'
else:
    data_file = None

dataset = pd.read_csv(data_file)

datasetNorm = (dataset - dataset.mean()) / (dataset.max() - dataset.min())

network_type = "RNN"
num_epochs = 100

batch_size = 1

total_series_length = len(dataset.index)
# TODO: change sequence for osu or tensor
truncated_backprop_length = 3 #The size of the sequence

state_size = 20 #The number of neurons

num_features = 5
num_classes = 5 #[1,0]

num_batches = total_series_length//batch_size//truncated_backprop_length

min_test_size = 100

print('The total series length is: %d' %total_series_length)
print('The current configuration gives us %d batches of %d observations each one looking %d steps in the past'
      %(num_batches,batch_size,truncated_backprop_length))

datasetTrain = datasetNorm[dataset.index < num_batches * batch_size * truncated_backprop_length]

test_first_idx = 0
for i in range(min_test_size, len(datasetNorm.index)):

    if (i % truncated_backprop_length * batch_size == 0):
        test_first_idx = len(datasetNorm.index) - i
        break

datasetTest = datasetNorm[dataset.index >= test_first_idx]

xTrain = datasetTrain[['new_files', 'deleted_files', 'modified_files', 'inserted_lines', 'deleted_lines']].as_matrix()
yTrain = datasetTrain[['new_files_pred', 'deleted_files_pred', 'modified_files_pred', 'inserted_lines_pred', 'deleted_lines_pred']].as_matrix()

xTest = datasetTest[['new_files', 'deleted_files', 'modified_files', 'inserted_lines', 'deleted_lines']].as_matrix()
yTest_pre = datasetTest[['new_files_pred', 'deleted_files_pred', 'modified_files_pred', 'inserted_lines_pred', 'deleted_lines_pred']]
yTest = yTest_pre.as_matrix()


batchX_placeholder = tf.placeholder(dtype=tf.float32,shape=[None,truncated_backprop_length,num_features],name='data_ph')
batchY_placeholder = tf.placeholder(dtype=tf.float32,shape=[None,truncated_backprop_length,num_classes],name='target_ph')

W2 = tf.Variable(initial_value=np.random.rand(state_size,num_classes),dtype=tf.float32)
b2 = tf.Variable(initial_value=np.random.rand(1,num_classes),dtype=tf.float32)
labels_series = tf.unstack(batchY_placeholder, axis=1)


if network_type == "RNN":
    cell = tf.contrib.rnn.BasicRNNCell(num_units=state_size)
elif network_type == "LSTM":
    cell = tf.contrib.rnn.BasicLSTMCell(num_units=state_size)
else:
    cell = None

states_series, current_state = tf.nn.dynamic_rnn(cell=cell,inputs=batchX_placeholder,dtype=tf.float32)
states_series = tf.transpose(states_series,[1,0,2])

last_state = tf.gather(params=states_series,indices=states_series.get_shape()[0]-1)
last_label = tf.gather(params=labels_series,indices=len(labels_series)-1)

weight = tf.Variable(tf.truncated_normal([state_size,num_classes]))
bias = tf.Variable(tf.constant(0.1,shape=[num_classes]))

prediction = tf.matmul(last_state,weight) + bias

loss = tf.reduce_mean(tf.squared_difference(last_label,prediction))

train_step = tf.train.AdamOptimizer(learning_rate=0.01).minimize(loss)

loss_list = []
test_pred_list = []

with tf.Session() as sess:
    tf.global_variables_initializer().run()

    for epoch_idx in range(num_epochs):

        print('Epoch %d' % epoch_idx)

        for batch_idx in range(num_batches):
            start_idx = batch_idx * truncated_backprop_length
            end_idx = start_idx + truncated_backprop_length * batch_size

            batchX = xTrain[start_idx:end_idx, :].reshape(batch_size, truncated_backprop_length, num_features)
            batchY = yTrain[start_idx:end_idx].reshape(batch_size, truncated_backprop_length, 5)

            feed = {batchX_placeholder: batchX, batchY_placeholder: batchY}

            # TRAIN!
            _loss, _train_step, _pred, _last_label, _prediction = sess.run(
                fetches=[loss, train_step, prediction, last_label, prediction],
                feed_dict=feed
            )

            loss_list.append(_loss)

            if (batch_idx % 1000 == 0):
                print('Step %d - Loss: %.6f' % (batch_idx, _loss))

    # TEST
    for test_idx in range(len(xTest) - truncated_backprop_length):
        testBatchX = xTest[test_idx:test_idx + truncated_backprop_length, :].reshape(
            (1, truncated_backprop_length, num_features))
        testBatchY = yTest[test_idx:test_idx + truncated_backprop_length].reshape((1, truncated_backprop_length, 5))
        # _current_state = np.zeros((batch_size,state_size))
        feed = {batchX_placeholder: testBatchX,
                batchY_placeholder: testBatchY}
        # Test_pred contains 'window_size' predictions, we want the last one
        _last_state, _last_label, test_pred = sess.run([last_state, last_label, prediction], feed_dict=feed)
        test_pred_list.append(test_pred[0])  # The last one


plt.title("{} repo. Loss of {} with layer size {}, sequence {}, {} epochs".format(repository, network_type,
                                                                                  state_size, truncated_backprop_length,
                                                                                  num_epochs))
plt.scatter(x=np.arange(0,len(loss_list)),y=loss_list)
plt.xlabel('epochs')
plt.ylabel('loss')
plt.show()

predicted = pd.DataFrame(test_pred_list, columns=['pred_new', 'pred_del', 'pred_mod', 'pred_insert', 'pred_del_lines'])

print("len of yTest is: {}\nlen of predicted is: {}".format(len(yTest_pre['modified_files_pred']), len(predicted['pred_mod'])))

predicted.columns = ['new_file_prediction', 'deleted_file_prediction', 'modified_file_prediction',
                     'inserted_lines_prediction', 'deleted_lines_prediction']
yTest_pre.columns = ['new_file_expected', 'deleted_file_expected', 'modified_file_expected',
                     'inserted_lines_expected', 'deleted_lines_expected']
yTest_pre['x'] = list(range(1, len(yTest_pre.index) + 1))
predicted['x'] = list(range(1, len(predicted.index) + 1))

make_plots(yTest_pre, predicted, num_epochs, truncated_backprop_length, state_size, network_type, repository)