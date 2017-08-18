
class UnmatchedLengthError(Exception):
    def __init__(self, **kwargs):
        temp = list()
        for key, value in kwargs.items():
            temp.append("{}: {}".format(key, value))
        super().__init__("Lengths don't match. {}".format(" ".join(temp)))

class LayerError(Exception):
    def __init__(self, mess=None):
        super().__init__(mess)
