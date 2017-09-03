import random


def get_rand():
    return random.random()


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
