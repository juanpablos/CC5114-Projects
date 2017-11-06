import random

from src.genetic_algorithm import GA
from src.hw1.NeuralNetwork import NeuralNetwork
from src.hw1.Utils.utilities import get_seeds, make_dataset, generate_network, get_rand

seed_train_set, seed_test_set, seed_train_expected, seed_test_expected = get_seeds("formatted_seeds.txt", 35)
epoch = 200
layers = 2
neurons = 20


# check for features and classes
def get_classes(data):
    classes = list()
    for line in data:
        if line[-1] not in classes:
            classes.append(line[-1])
    return len(classes)


topology = (len(seed_train_set[0]), get_classes(seed_train_expected))  # in | out


def network_generator():
    return generate_network(topology[0], topology[1], neurons, layers)


def weight_swapper(f, p1, p2, mutation):
    layer = random.randint(len(p1))
    neuron = random.randint(len(layer))
    weight = random.randint(len())


def fitness_function_weights(network):
    the_network = NeuralNetwork(manual=True)
    the_network.initialize(network=network)
    correctness = 0.
    for t, ex in zip(seed_test_set, seed_test_expected):
        ideal_output = ex
        actual = the_network.feed(t)
        normalized_output = [round(e, 0) for e in actual]
        if ideal_output == normalized_output:
            correctness += 1.
    return correctness / len(seed_test_set)


if __name__ == '__main__':
    generators = []
    # genes are layers
    ga = GA(pop_size=1000, mutation_rate=0.01, genes=layers, fitness=fitness_function_weights,
            net_generator=network_generator, generators=generators,
            min_fitness=0.9, max_iter=1000)
