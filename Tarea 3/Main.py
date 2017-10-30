import numpy as np

data = list()


# check for features and classes
def get_classes(data):
    classes = list()
    for line in data:
        if line[-1] not in classes:
            classes.append(line[-1])
    return len(classes)


topology = (len(data[0]), get_classes(data))  # in | out


def generator():
    # check
    return np.random.rand(topology[0])


def fitness(networks):
    fitness_list = list()
    for network in networks:
        fitness_list.append(network.eval(data))
    return fitness_list
