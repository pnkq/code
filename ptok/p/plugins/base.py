from abc import ABC, abstractmethod


class TokenizerPlugin(ABC):
    @abstractmethod
    def accepts(self, lang):
        pass

    @abstractmethod
    def tokenize(self, span, return_pieces):
        pass

