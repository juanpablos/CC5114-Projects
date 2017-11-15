import random


class Node:
    def __init__(self, value, left=None, right=None):
        self.value = value
        self.l = left
        self.r = right

    def eval(self):
        pass


class InnerNode(Node):
    def eval(self):
        return self.value(self.l.eval(), self.r.eval())

    def __str__(self):
        return "{} {} {}".format(str(self.l), self.value.__name__, str(self.r))


class TerminalNode(Node):
    def eval(self):
        return self.value

    def __str__(self):
        return str(self.value)


class NullNode(Node):
    def eval(self):
        pass


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
    a = TerminalNode(3)
    b = TerminalNode(4)
    c = InnerNode(lambda x, y: x + y, a, b)
    d = TerminalNode(10)
    e = InnerNode(lambda x, y: x * y, c, d)

    print(e.eval())


    @rename("+")
    def add(x, y):
        return x + y


    @rename("-")
    def sub(x, y):
        return x - y


    f = [add, sub]
    t = [1, 2, 3]
    ast = AST(f, t, 2)

    tree = ast.create_tree()
    print(tree)
    print(tree.eval())
