import csv
import random
import sys

import numpy as np

from src.gen_prog import GP


def rename(new_name):
    def decorator(f):
        f.__name__ = new_name
        return f

    return decorator


@rename("({0} + {1})")
def add(x, y):
    return x + y


@rename("({0} - {1})")
def sub(x, y):
    return x - y


@rename("({0} * {1})")
def mult(x, y):
    return x * y


@rename("({0} / {1})")
def div(x, y):
    try:
        return x * y
    except ZeroDivisionError:
        return sys.maxsize


if __name__ == "__main__":
    # only used when working with numbers
    target_number = 5649163


    def fitness(element):
        generated = element.eval()

        return abs(target_number - generated)


    def length_fitness(element):
        return fitness(element) + len(element.serialize())


    def fitness2(element):
        def func(x):
            return 5 * x * x + 10 * x + 26

        error = 0.
        ec = func
        for n in np.arange(-10., 11, 1):
            error += abs(ec(n) - element.eval({'x': n}))
        return error + float(len(element.serialize()))


    # function set
    functions = [add, sub, mult, div]
    # fitness function to use
    the_fitness = length_fitness
    # equations
    # terminals = random.sample(list(np.arange(-10, 11, 1)), 10) + ['x'] * 10
    # numbers
    terminals = random.sample(range(100), 10)
    population = 100
    depth = 5
    crossover_rate = 0.9
    mutation_rate = 0.01
    iterations = 100
    min_fitness = 0

    gp = GP(terminal_set=terminals, function_set=functions, fitness=the_fitness, pop_size=population, depth=depth,
            crossover_rate=crossover_rate, mutation_rate=mutation_rate, iterations=iterations, min_fitness=min_fitness)

    fitness_evolution, average_fitness_evolution, best = gp.run()

    print("best tree is: {}".format(str(best)))
    print("fitness {}".format(the_fitness(best)))
    print(fitness_evolution)

    result_dir = "Results/"
    res = "1"

    with open(result_dir + "gp_out_{}.csv".format(res), 'w', newline="\n") as o:
        out = csv.writer(o)
        out.writerow(['generation', 'best_fitness', 'avg_value'])
        for i, fit in enumerate(fitness_evolution):
            out.writerow([i, fit, average_fitness_evolution[i]])

    with open(result_dir + "gp_out_{}_info.txt".format(res), 'w', newline="\n") as o:
        o.write("target number: {}\n".format(target_number))
        o.write("best tree: {}\n".format(str(best)))
        o.write("terminals: {}\n".format(terminals))
        o.write("population: {}\n".format(population))
        o.write("initial depth: {}\n".format(depth))
        o.write("crossover rate: {}\n".format(crossover_rate))
        o.write("mutation rate: {}\n".format(mutation_rate))
        o.write("max iterations: {}\n".format(iterations))
        o.write("min fitness: {}\n".format(min_fitness))
