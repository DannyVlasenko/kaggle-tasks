import functools


class Composable:

    def __init__(self, func):
        self.func = func
        functools.update_wrapper(self, func)

    def __mul__(self, other):
        return lambda *args, **kw: self.func(other(*args, **kw))

    def __call__(self, *args, **kw):
        return self.func(*args, **kw)
