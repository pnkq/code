from p.plugins.vie import VietnamesePlugin
from p.plugins.eng import EnglishPlugin
from p.plugins.unk import UnknownPlugin
from p.piece import Piece


class PluginManager:

    def __init__(self):
        self.plugins = [
            VietnamesePlugin(),
            EnglishPlugin(),
            UnknownPlugin()
        ]

    def process(self, tokens):
        word_id = 0
        for token in tokens:
            for plugin in self.plugins:
                if plugin.accepts(token.lang):
                    parts = plugin.tokenize(token, True)
                    for part in parts:
                        part.word_id = word_id
                        yield part
                    word_id += 1
                    break
    
    def process_pairs(self, pairs):
        for pair in pairs:
            for plugin in self.plugins:
                if plugin.accepts(pair[1]):
                    parts = plugin.tokenize(pair, False)
                    for part in parts:
                        yield part
                    break


    
