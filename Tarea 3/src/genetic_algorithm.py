import random

import numpy as np


class GA:
    def __init__(self, pop_size, mutation_rate, fitness, net_generator, breed_function, single_gen, min_fitness,
                 max_iter=100):
        self.population_size = pop_size
        self.mutation = mutation_rate
        self.fitness_function = fitness
        self.net_generator = net_generator
        self.breed_function = breed_function
        self.max_iterations = max_iter
        self.min_fitness = min_fitness
        self.select_function = self.select
        self.random_generator = single_gen
        self.population = list()
        self.fitness = list()

    def run(self):
        i = 0
        self.population = self.generate_population()
        self.fitness = self.get_fitness(self.population)
        best = self.population[self.fitness.index(max(self.fitness))]
        while i <= self.max_iterations and self.fitness_function(best) < self.min_fitness:
            print("iter {} of {}".format(i, self.max_iterations))
            print("best is: {}\nwith {} acc. Avg: {}".format(best, max(self.fitness), np.mean(self.fitness)))
            n_fitness = self.normalize(self.fitness)
            parents = self.select_function(self.population, n_fitness)
            self.population = self.create_new_population(parents)
            self.fitness = self.get_fitness(self.population)
            i += 1
            best = self.population[self.fitness.index(max(self.fitness))]
        print("{} generations".format(i))
        last_fitness = self.get_fitness(self.population)
        best = self.population[last_fitness.index(max(last_fitness))]

        return best, i - 1

    def generate_population(self):
        population = list()
        for p in range(self.population_size):
            population.append(self.net_generator())
        return population

    def get_fitness(self, population):
        fitness = list()
        for p in population:
            fitness.append(self.fitness_function(p))
        return fitness

    def normalize(self, fitness):
        the_sum = sum(fitness)
        if the_sum != 0:
            return [f / the_sum for f in fitness]
        else:
            return fitness

    def select(self, population, n_fitness):
        ordered_pair = [(x, y) for x, y in sorted(zip(population, n_fitness), key=lambda pair: pair[1], reverse=True)]
        acc_fitness = [ordered_pair[0][1]]
        for i in range(1, len(ordered_pair)):
            acc_fitness.append(acc_fitness[i - 1] + ordered_pair[i][1])
        parents = list()
        for _ in range(self.population_size):
            index1 = self.get_random(acc_fitness)
            index2 = self.get_random(acc_fitness)
            parents.append(ordered_pair[index1][0])
            parents.append(ordered_pair[index2][0])
        return parents

    def create_new_population(self, parents):
        i = 0
        new_population = list()
        while i < len(parents):
            p1 = parents[i]
            p2 = parents[i + 1]
            i += 2
            new_population.append(self.create_new_element(p1, p2))
        return new_population

    def get_random(self, acc_fitness):
        n = random.random()
        if n < acc_fitness[0]:
            return 0
        for i in range(1, len(acc_fitness) - 1):
            if acc_fitness[i] > n:
                return i - 1
        return len(acc_fitness) - 1

    def create_new_element(self, p1, p2):
        child1, child2 = self.breed_function(p1, p2)
        final_child = child1
        candidates = [child1, child2, p1, p2]
        fitness = [self.fitness_function(child1), self.fitness_function(child2),
                   self.fitness[self.population.index(p1)], self.fitness[self.population.index(p2)]]

        if max(fitness[0], fitness[1]) < (max(fitness[2], fitness[3]) * 0.8):
            # if the children are too weak, keep the strongest parent
            final_child = candidates[fitness.index(max(fitness[2], fitness[3]))]

        # mutation
        if random.random() < self.mutation:
            final_child = self.mutate(final_child)

        return final_child

    def mutate(self, final_child):
        rand = random.random()
        layer = random.randrange(len(final_child))
        neuron = random.randrange(len(final_child[layer]))

        # weights
        if rand < 0.8:
            weight = random.randrange(len(final_child[layer][neuron]))
            rand = random.random()
            # scale weight
            if rand < 0.5:
                final_child[layer][neuron][weight] *= random.uniform(-2, 2)
                # replace
            else:
                final_child[layer][neuron][weight] = self.random_generator()

        # neuron
        else:
            n_weights = len(final_child[layer][neuron])
            for w in range(n_weights):
                final_child[layer][neuron][w] = self.random_generator()

        return final_child
