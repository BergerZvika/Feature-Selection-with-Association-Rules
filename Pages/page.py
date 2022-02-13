import abc
from abc import ABC


class Page(ABC):
    def __init__(self):
        pass

    @abc.abstractmethod
    def show_page(self):
        raise NotImplementedError
