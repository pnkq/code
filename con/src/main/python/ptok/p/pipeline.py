from p.normalizer import Normalizer
from p.dispatcher import Dispatcher
from p.plugin_manager import PluginManager
from p.postprocessor import PostProcessor
from p.piece import Piece


class Pipeline:

    def __init__(self):
        self.normalizer = Normalizer()
        self.dispatcher = Dispatcher()
        self.plugins = PluginManager()
        self.postprocessor = PostProcessor()

    def tokenize(self, text) -> list[Piece]:
        text = self.normalizer.normalize(text)
        spans = self.dispatcher.dispatch(text)
        pieces = self.plugins.process(spans)
        pieces = self.postprocessor.process(pieces)
        return pieces

    def tokenize_to_text(self, text) -> list[str]:
        text = self.normalizer.normalize(text)
        pairs = self.dispatcher.dispatch_to_text_and_lang(text)
        subs = self.plugins.process_pairs(pairs)
        subs = self.postprocessor.process(subs)
        return subs

