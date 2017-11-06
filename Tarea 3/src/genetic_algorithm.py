import random


class GA:
    def __init__(self, pop_size, mutation_rate, genes, fitness, net_generator, breed_functions, min_fitness,
                 max_iter=100):
        self.population_size = pop_size
        self.mutation = mutation_rate
        self.genes = genes
        self.fitness_function = fitness
        self.net_generator = net_generator
        self.breed_functions = breed_functions
        self.max_iterations = max_iter
        self.min_fitness = min_fitness
        self.select_function = self.select

    def run(self):
        i = 0
        population = self.generate_population()
        fitness = self.get_fitness(population)
        best = population[fitness.index(max(fitness))]
        while i <= self.max_iterations and self.fitness_function(best) < self.min_fitness:
            print("iter {} of {}".format(i, self.max_iterations))
            n_fitness = self.normalize(fitness)
            parents = self.select_function(population, n_fitness)
            population = self.create_new_population(parents)
            fitness = self.get_fitness(population)
            i += 1
            best = population[fitness.index(max(fitness))]
        print("{} generations".format(i))
        last_fitness = self.get_fitness(population)
        best = population[last_fitness.index(max(last_fitness))]

        return best

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
        return [f / the_sum for f in fitness]

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
        breed = random.choice(self.breed_functions)
        return breed(self.fitness_function, p1, p2, self.mutation)
