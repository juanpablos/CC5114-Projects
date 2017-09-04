import csv
import random


def get_rand():
    res = 0.
    while res < 0.2:
        res = random.random()
    return res


def prettify_network(a_network):
    for layer in a_network:
        for neuron in layer:
            print(neuron)
        print("")


# generates a random network with the input parameters
def generate_network(n_inputs, neurons_output, neurons_per_layer, layers):
    assert layers > 0

    a_network = list()

    if layers < 2:
        first_last_layer = list()
        # input - output
        for out_n in range(neurons_output):
            input_out_neuron = list()
            for n_input in range(n_inputs + 1):
                input_out_neuron.append(get_rand())
            first_last_layer.append(input_out_neuron)
        a_network.append(first_last_layer)

    else:
        first_layer = list()
        # input
        # input is number of weights in initial layer
        for input_n in range(neurons_per_layer):
            input_neuron = list()
            for n_input in range(n_inputs + 1):
                input_neuron.append(get_rand())
            first_layer.append(input_neuron)
        a_network.append(first_layer)

        # rest
        for layer in range(layers - 2):
            a_layer = list()
            for neuron in range(neurons_per_layer):
                a_neuron = list()
                for _ in range(neurons_per_layer + 1):
                    a_neuron.append(get_rand())
                a_layer.append(a_neuron)
            a_network.append(a_layer)

        last_layer = list()
        # out
        # out is number of neurons in outer later
        for out_n in range(neurons_output):
            output_neuron = list()
            for n_out in range(neurons_per_layer + 1):
                output_neuron.append(get_rand())
            last_layer.append(output_neuron)
        a_network.append(last_layer)

    return a_network


def normalize(value, min_value, max_value, norm_l=0, norm_h=1):
    out = (value - min_value) * (norm_h - norm_l)
    out /= (max_value - min_value)
    out += norm_l
    return out


# Dont mind this...
def get_normalized_seeds(file):
    with open(file) as f:
        ff = csv.reader(f, delimiter=',')
        col1 = list()
        col2 = list()
        col3 = list()
        col4 = list()
        col5 = list()
        col6 = list()
        col7 = list()
        colexp = list()
        for row in ff:
            col1.append(float(row[0].strip('\t')))
            col2.append(float(row[1].strip('\t')))
            col3.append(float(row[2].strip('\t')))
            col4.append(float(row[3].strip('\t')))
            col5.append(float(row[4].strip('\t')))
            col6.append(float(row[5].strip('\t')))
            col7.append(float(row[6].strip('\t')))
            colexp.append(row[7].strip('\t'))
        minmax1 = [min(col1), max(col1)]
        minmax2 = [min(col2), max(col2)]
        minmax3 = [min(col3), max(col3)]
        minmax4 = [min(col4), max(col4)]
        minmax5 = [min(col5), max(col5)]
        minmax6 = [min(col6), max(col6)]
        minmax7 = [min(col7), max(col7)]

        _file = list()

        for i in range(len(col1)):
            row = list()
            row.append(normalize(col1[i], minmax1[0], minmax1[1]))
            row.append(normalize(col2[i], minmax2[0], minmax2[1]))
            row.append(normalize(col3[i], minmax3[0], minmax3[1]))
            row.append(normalize(col4[i], minmax4[0], minmax4[1]))
            row.append(normalize(col5[i], minmax5[0], minmax5[1]))
            row.append(normalize(col6[i], minmax6[0], minmax6[1]))
            row.append(normalize(col7[i], minmax7[0], minmax7[1]))

            if colexp[i] == '1':
                exp = [1, 0, 0]
            elif colexp[i] == '2':
                exp = [0, 1, 0]
            else:
                exp = [0, 0, 1]

            _file.append([row, exp])

        return _file


# Just works for the seeds dataset, but the idea is to divide the dataset
def split_seeds(formatted_array, number_test=40):
    train_set = list()
    train_expected = list()
    test_set = list()
    test_expected = list()

    counter = 0
    # assumes the classes are sorted
    for row in range(len(formatted_array)):

        if test_expected and test_expected[len(test_expected) - 1] != formatted_array[row][1]:
            counter = 0

        if counter >= number_test:
            arg, exp = formatted_array[row]
            train_set.append(arg)
            train_expected.append(exp)
        else:
            arg, exp = formatted_array[row]
            test_set.append(arg)
            test_expected.append(exp)

        counter += 1

    return train_set, test_set, train_expected, test_expected


def get_seeds(seeds_file, test_sample_class):
    return split_seeds(get_normalized_seeds(seeds_file), test_sample_class)


def make_dataset(data, expected):
    dataset = list()
    for d in zip(data, expected):
        dataset.append(list(d))
    return dataset
