
class UnmatchedLengthError(Exception):
    def __init__(self, **kwargs):
        super().__init__("Lengths don't match.")
        self.error_args = kwargs