from p.plugins.vie import VietnamesePlugin
from p.plugins.eng import EnglishPlugin
from p.plugins.unk import UnknownPlugin


class PluginManager:

    def __init__(self):
        self.plugins = [
            VietnamesePlugin(),
            EnglishPlugin(),
            UnknownPlugin()
        ]

    def process(self, spans):
        pieces = []
        word_id = 0
        for span in spans:
            for plugin in self.plugins:
                if plugin.accepts(span):
                    parts = plugin.tokenize(span)
                    for part in parts:
                        part.word_id = word_id
                    pieces.extend(parts)
                    word_id += 1
                    break
        return pieces
    
