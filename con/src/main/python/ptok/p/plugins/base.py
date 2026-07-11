from abc import ABC, abstractmethod


class TokenizerPlugin(ABC):
    @abstractmethod
    def accepts(self, span):
        pass

    @abstractmethod
    def tokenize(self, span):
        pass

