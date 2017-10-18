import random
import string


class GA:
    def __init__(self, pop_size, mutation_rate, genes, fitness, generator, max_iter, select='default'):
        self.population_size = pop_size
        self.mutation = mutation_rate
        self.genes = genes
        self.fitness_function = fitness
        self.generator_function = generator
        self.max_iterations = max_iter

        if select == 'default':
            self.select_function = self.select_default
        else:
            self.select_function = self.select_roulette


    def run(self):
        i = 0
        population = self.generate_population()
        while i <= self.max_iterations:
            print("iter {} of {}".format(i, self.max_iterations))
            n_fitness = self.normalize(self.get_fitness(population))
            parents = self.select_function(population, n_fitness)
            population = self.create_new_population(parents)
            i += 1
        print("{} generations".format(i))
        last_fitness = self.get_fitness(population)
        best = population[last_fitness.index(max(last_fitness))]

        return best

    def generate_population(self):
        population = list()
        for p in range(self.population_size):
            ind = []
            for g in range(self.genes):
                ind.append(self.generator_function())
            population.append(ind)
        return population

    def get_fitness(self, population):
        fitness = list()
        for p in population:
            fitness.append(self.fitness_function(p))
        return fitness

    def normalize(self, fitness):
        the_sum = sum(fitness)
        return [f / the_sum for f in fitness]

    def select_default(self, population, n_fitness):
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

    def select_roulette(self, population, n_fitness):
        parents = list()
        for _ in range(2 * self.population_size):
            prob = random.uniform(0, sum(n_fitness))
            index = 0
            for i, fitness in enumerate(n_fitness):
                if prob <= 0:
                    index = i
                    break
                prob -= fitness
            parents.append(population[index])
        return parents

    def create_new_element(self, p1, p2):
        index = random.randrange(self.genes)
        new_element = p1[:index] + p2[index:]
        for i in range(len(new_element)):
            if random.random() <= self.mutation:
                new_element[i] = self.generator_function()
        return new_element



if __name__ == '__main__':
    def f(x):
        correct = 'hellohello'
        res = 0
        for i, c in zip(x, correct):
            if i == c:
                res += 1
        return res

    def g1():
        r = random.random()
        if r <= 0.5:
            return '1'
        else:
            return '0'

    def g2():
        return random.choice(string.ascii_lowercase)


    ga = GA(100, 0.05, 10, f, g2, max_iter=100, select='default')
    print(ga.run())