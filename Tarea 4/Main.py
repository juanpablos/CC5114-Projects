import random

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


def fitness(element):
    number = 5184618
    generated = element.eval()

    return abs(number - generated)


if __name__ == "__main__":
    functions = [add, sub, mult]
    terminals = random.sample(range(1000), 20)
    gp = GP(terminal_set=terminals, function_set=functions, fitness=fitness, pop_size=100, depth=3, crossover_rate=0.9,
            mutation_rate=0.01, iterations=100, min_fitness=0)

    fitness_evolution, best = gp.run()
    print("-" * 20)

    print("best tree is: {}".format(str(best)))
    print("fitness {}".format(fitness(best)))
    print("value {}".format(best.eval()))

    print("-" * 20)
    print(terminals)
    print(fitness_evolution)
