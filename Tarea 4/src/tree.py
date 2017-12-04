import copy
import random


class Node:
    def __init__(self, value, left=None, right=None):
        self.value = value
        self.l = left
        self.r = right

    def eval(self, eval_dict=None):
        pass

    def serialize(self):
        pass

    def copy(self):
        return copy.deepcopy(self)

    def replace(self, node):
        # TODO: check that node is instance of Node
        self.__class__ = node.__class__
        self.l = node.l
        self.r = node.r
        self.value = node.value


class InnerNode(Node):
    def __init__(self, value, left, right):
        super().__init__(value, left, right)

    def __str__(self):
        return self.value.__name__.format(str(self.l), str(self.r))

    def eval(self, eval_dict=None):
        return self.value(self.l.eval(eval_dict), self.r.eval(eval_dict))

    def serialize(self):
        return self.l.serialize() + [self] + self.r.serialize()


class TerminalNode(Node):
    def __init__(self, value):
        super().__init__(value)

    def __str__(self):
        return str(self.value)

    def eval(self, eval_dict=None):
        try:
            return eval_dict[self.value]
        except (KeyError, TypeError):
            return self.value

    def serialize(self):
        return [self]


class AST:
    def __init__(self, functions, terminals, depth):
        self.functions = functions
        self.terminals = terminals
        self.depth = depth

    def create(self, max_depth=None):
        def create_rec_tree(depth):
            if depth > 0:
                return InnerNode(random.choice(self.functions), create_rec_tree(depth - 1), create_rec_tree(depth - 1))
            else:
                return TerminalNode(random.choice(self.terminals))

        # python has short-circuit boolean expressions AKA lazy conditions
        return create_rec_tree(max_depth or self.depth)


def rename(new_name):
    def decorator(f):
        f.__name__ = new_name
        return f

    return decorator


if __name__ == "__main__":
    @rename("({0} + {1})")
    def add(x, y):
        return x + y


    @rename("({0} - {1})")
    def sub(x, y):
        return x - y


    @rename("({0} * {1})")
    def mult(x, y):
        return x * y


    @rename("max({0}, {1})")
    def _max(x, y):
        return max(x, y)


    funs = [add, sub, mult, _max]
    t = [i for i in range(5)] + ['x']
    ast = AST(funs, t, 2)

    tree = ast.create()
    print(tree)
    print(tree.eval({'x': 10}))
