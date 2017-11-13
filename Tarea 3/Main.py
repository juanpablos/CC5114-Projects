import csv
import random

from src.genetic_algorithm import GA
from src.hw1.NeuralNetwork import NeuralNetwork
from src.hw1.Utils.utilities import get_seeds, generate_network

seed_train_set, seed_test_set, seed_train_expected, seed_test_expected = get_seeds("formatted_seeds.txt", 35)
test_data = seed_train_set + seed_test_set
test_expected = seed_train_expected + seed_test_expected
layers = 2
neurons = 5

topology = (len(seed_train_set[0]), len(seed_train_expected[0]))  # in | out


def network_generator():
    return generate_network(topology[0], topology[1], neurons, layers)


def get_rand():
    return random.uniform(-5, 5)


def breed(p1, p2):
    layer = random.randrange(len(p1))
    neuron = random.randrange(len(p1[layer]))

    child1 = p1[:layer] + p2[layer:]
    child2 = p2[:layer] + p1[layer:]

    for n in range(neuron):
        child1[layer][n], child2[layer][n] = child2[layer][n], child1[layer][n]

    return child1, child2


def fitness_function_weights(network):
    the_network = NeuralNetwork(manual=True)
    the_network.initialize(network=network)
    correctness = 0.
    for t, ex in zip(test_data, test_expected):
        ideal_output = ex
        actual = the_network.feed(t)
        normalized_output = [round(e, 0) for e in actual]
        if ideal_output == normalized_output:
            correctness += 1.
    return correctness / len(test_data)


if __name__ == '__main__':
    ga = GA(pop_size=500, mutation_rate=0.01, fitness=fitness_function_weights,
            net_generator=network_generator, single_gen=get_rand, breed_function=breed,
            min_fitness=0.9, max_iter=100)
    fitness, avg, best = ga.run()
    print("best is: {}\nwith {} acc\ntook {} generations".format(best, fitness[-1], len(fitness)))

    with open("network_out2.csv", 'w', newline="\n") as o:
        out = csv.writer(o)
        out.writerow(['generation', 'best_fitness', 'avg_value'])
        for i, fit in enumerate(fitness):
            out.writerow([i, fit, avg[i]])