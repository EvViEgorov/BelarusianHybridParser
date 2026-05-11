from . import morph
from .morph import MorphAnalyzer


class BeHybridParser:
    def __init__(self, text):
        self.text = text
        self.morph = MorphAnalyzer()
        self.analysis = self._analyze_text(text)

    def _tokenize_text(self, text):
        tokenized = text.split(' ')
        return tokenized

    def _analyze_text(self, text):
        analyses = []
        for token in self._tokenize_text(text):
            analysis = self.morph.analyze_word(token)
            analyses.append(analysis)
        return analyses