import copy
import random


class Node:
    def __init__(self, value, left=None, right=None):
        self.value = value
        self.l = left
        self.r = right

    def eval(self):
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
    def __str__(self):
        return "({} {} {})".format(str(self.l), self.value.__name__, str(self.r))

    def eval(self):
        return self.value(self.l.eval(), self.r.eval())

    def serialize(self):
        return self.l.serialize() + [self] + self.r.serialize()


class TerminalNode(Node):
    def __str__(self):
        return str(self.value)

    def eval(self):
        return self.value

    def serialize(self):
        return [self]


class AST:
    def __init__(self, functions, terminals, depth):
        self.functions = functions
        self.terminals = terminals
        self.depth = depth

    def create_tree(self):
        def create_rec_tree(depth):
            if depth > 0:
                return InnerNode(random.choice(self.functions), create_rec_tree(depth - 1), create_rec_tree(depth - 1))
            else:
                return TerminalNode(random.choice(self.terminals))

        return create_rec_tree(self.depth)


def rename(new_name):
    def decorator(f):
        f.__name__ = new_name
        return f

    return decorator


if __name__ == "__main__":
    @rename("+")
    def add(x, y):
        return x + y


    @rename("-")
    def sub(x, y):
        return x - y


    @rename("*")
    def mult(x, y):
        return x * y


    funs = [add, sub, mult]
    t = [i for i in range(50)]
    ast = AST(funs, t, 2)

    tree = ast.create_tree()
    print(tree)
    print(tree.eval())
