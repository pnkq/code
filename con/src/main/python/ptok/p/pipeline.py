from p.normalizer import Normalizer
from p.dispatcher import Dispatcher
from p.plugin_manager import PluginManager
from p.postprocessor import PostProcessor


class Pipeline:

    def __init__(self):
        self.normalizer = Normalizer()
        self.dispatcher = Dispatcher()
        self.plugins = PluginManager()
        self.postprocessor = PostProcessor()

    # def tokenize(self, text):
    #     text = self.normalizer.normalize(text)
    #     spans = self.dispatcher.dispatch(text)
    #     pieces = self.plugins.process(spans)
    #     pieces = self.postprocessor.process(pieces)
    #     return pieces

    def encode(self, text):
        text = self.normalizer.normalize(text)
        spans = self.dispatcher.dispatch(text)
        pieces = self.plugins.process(spans)
        return self.encoder.encode(text, pieces)
