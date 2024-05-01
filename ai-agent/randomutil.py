import random
from datetime import datetime


class Random:
    def __init__(self, seed=None):
        if seed is None:
            self.seed = datetime.now().timestamp()
        else:
            self.seed = seed

        random.seed(self.seed)
        self.random = random.random
        self.randrange = random.randrange

    def fraction(self):
        return self.random()

    def range(self, stop):
        return self.randrange(stop)
