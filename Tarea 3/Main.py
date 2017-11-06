import random

from src.hw1.Utils.utilities import get_seeds

seed_train_set, seed_test_set, seed_train_expected, seed_test_expected = get_seeds("formatted_seeds.txt", 35)


# check for features and classes
def get_classes(data):
    classes = list()
    for line in data:
        if line[-1] not in classes:
            classes.append(line[-1])
    return len(classes)


topology = (len(seed_train_set[0]), get_classes(seed_train_expected))  # in | out


def generator_hyperparameters():
    layer_range = list(range(1, 5))
    neuron_number_range = list(range(2, 211, 10))
    learning_rate_range = [i / 100 for i in range(1, 60, 10)]

    return {'n_inputs': topology[0], 'neurons_output': topology[1],
            'layers': random.choice(layer_range), 'neurons_per_layer': random.choice(neuron_number_range),
            'learning_rate': random.choice(learning_rate_range)}
