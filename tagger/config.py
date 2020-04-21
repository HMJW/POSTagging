# -*- coding: utf-8 -*-

from ast import literal_eval
from configparser import ConfigParser


class Config(object):

    def __init__(self, path):
        super(Config, self).__init__()

        self.config = ConfigParser()
        self.config.read(path)
        self.kwargs = dict((option, literal_eval(value))
                           for section in self.config.sections()
                           for option, value in self.config.items(section))

    def __repr__(self):
        s = "-" * 15 + "-+-" + "-" * 25 + "\n"
        s += f"{'Param':15} | {'Value':^25}\n"
        s += "-" * 15 + "-+-" + "-" * 25 + "\n"
        for i, (option, value) in enumerate(self.kwargs.items()):
            s += f"{option:15} | {str(value):^25}\n"
        s += "-" * 15 + "-+-" + "-" * 25 + "\n"

        return s

    def __getattr__(self, attr):
        return self.kwargs.get(attr, None)

    def __getstate__(self):
        return vars(self)

    def __setstate__(self, state):
        self.__dict__.update(state)

    def update(self, kwargs):
        self.kwargs.update(kwargs)
