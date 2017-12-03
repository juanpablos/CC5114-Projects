import random
import sys

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

    def run(self):
        # FIXME: optimise -> multiple unneeded fitness executions
        i = 0
        best_fitness_list = list()
        population = self.generate_population()
        fitness, real_fitness = self.get_fitness(population)
        best = population[fitness.index(max(fitness))]
        while i < self.max_iterations and self.fitness_function(best) > self.min_fitness:
            print("iteration {} of {}".format(i, self.max_iterations))
            best_fitness_list.append(min(real_fitness))
            # TODO: assert fitness.index(max(fitness)) == real_fitness.index(min(real_fitness))
            print("best is: {}\nwith {} fitness".format(best.eval(), best_fitness_list[-1]))
            n_fitness = self.normalize(fitness)
            parents = self.select(population, n_fitness)
            population = self.create_new_population(parents)
            fitness, real_fitness = self.get_fitness(population)
            i += 1
            best = population[fitness.index(max(fitness))]
        print("iteration {} of {}".format(i, self.max_iterations))
        last_fitness, last_real_fitness = self.get_fitness(population)
        best = population[last_fitness.index(max(last_fitness))]
        best_fitness_list.append(min(last_real_fitness))
        print("best is: {}\nwith {} fitness".format(best.eval(), best_fitness_list[-1]))
        return best_fitness_list, best

    def generate_population(self):
        population = list()
        for _ in range(self.population_size):
            population.append(self.generator.create())
        return population

    def get_fitness(self, population):
        fitness = list()
        real_fitness = list()
        for p in population:
            real_f = self.fitness_function(p)
            real_fitness.append(real_f)
            try:
                f = 1. / real_f
            except ZeroDivisionError:
                f = sys.maxsize
            fitness.append(f)
        return fitness, real_fitness

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
        rand = random.random()
        if rand <= self.crossover_rate:
            new_element = p1.copy()
            # one liner because why not :)
            # selects a random node in the copy of parent 1, then replaces it with a copy of a random node in parent 2
            random.choice(new_element.serialize()).replace(random.choice(p2.serialize()).copy())
        else:
            if self.fitness_function(p1) < self.fitness_function(p2):
                new_element = p2.copy()
            else:
                new_element = p1.copy()

        rand = random.random()
        if rand <= self.mutation_rate:
            # a new subtree with random depth in [0, max_depth]
            depth = random.randint(0, self.generator.depth)
            random.choice(new_element.serialize()).replace(self.generator.create(depth))

        return new_element
