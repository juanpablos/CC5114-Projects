import random

from src.tree import AST


class GP:
    def __init__(self, terminal_set, function_set, fitness, pop_size=500, depth=3, crossover_rate=0.9,
                 mutation_rate=0.01, iterations=50, min_fitness=0):
        self.generator = AST(function_set, terminal_set, depth)
        self.fitness_function = fitness
        self.population_size = pop_size
        self.crossover_rate = crossover_rate
        self.mutation_rate = mutation_rate
        self.max_iterations = iterations
        self.min_fitness = min_fitness

    def generate_population(self):
        population = list()
        for _ in range(self.population_size):
            population.append(self.generator.create())
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
        # TODO: don't double call
        for _ in range(self.population_size):
            index1 = self.get_random(acc_fitness)
            index2 = self.get_random(acc_fitness)
            parents.append(ordered_pair[index1][0])
            parents.append(ordered_pair[index2][0])
        return parents

    def get_random(self, acc_fitness):
        n = random.random()
        if n <= acc_fitness[0]:
            return 0
        for i in range(1, len(acc_fitness) - 1):
            if acc_fitness[i] > n:
                return i - 1
        return len(acc_fitness) - 1

    def create_new_population(self, parents):
        i = 0
        new_population = list()
        while i < len(parents):
            # TODO: check in other way
            p1 = parents[i]
            p2 = parents[i + 1]
            i += 2
            new_population.append(self.create_new_element(p1, p2))

        return new_population

    def create_new_element(self, p1, p2):
        new_element = None
        rand = random.random()
        if rand <= self.crossover_rate:
            # crossover
            pass
        else:
            # return best of the 2
            pass

        rand = random.random()
        if rand <= self.mutation_rate:
            # mutate
            pass

        return new_element
