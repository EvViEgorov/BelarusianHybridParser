import stanza
from stanza.resources.common import ResourcesFileNotFoundError
from .morph import MorphAnalyzer


class BeHybridParser:
    _stanza_pipeline = None

    def __init__(self, text, neural_hints=None, use_dicts=True):
        if neural_hints not in (None, 'lemma', 'pos_feats'):
            raise ValueError("neural_hints takes only the following arguments: None, 'lemma', 'pos_feats'")
        self.text = text
        self._neural_hints = neural_hints
        self._use_dicts = use_dicts
        self._morph = MorphAnalyzer()
        BeHybridParser._init_stanza()
        self._token_info = self._tokenize_text(text)
        self.tokens = [t['text'] for t in self._token_info]
        self.analysis = self._analyze_text(self._token_info)

    @classmethod
    def _init_stanza(cls):
        if cls._stanza_pipeline is None:
            try:
                cls._stanza_pipeline = stanza.Pipeline(
                    lang='be',
                    processors='tokenize,pos,lemma',
                    download_method=None
                )
            except (ResourcesFileNotFoundError, FileNotFoundError):
                print("Stanza model 'be' not found. Downloading...")
                stanza.download('be')
                cls._stanza_pipeline = stanza.Pipeline(
                    lang='be',
                    processors='tokenize,pos',
                    download_method=None
                )

    def _tokenize_text(self, text):
        doc = self._stanza_pipeline(text)
        result = []
        for sentence in doc.sentences:
            for token in sentence.tokens:
                word = token.words[0]
                result.append({
                    'text': token.text,
                    'lemma': word.lemma,
                    'upos': word.upos,
                    'feats': word.feats if word.feats else ''
                })
        return result

    def _analyze_text(self, token_info_list):
        analyses = []
        for info in token_info_list:
            lemma_hint = None
            pos_hint = None
            feats_hint = None

            if self._neural_hints == 'lemma':
                lemma_hint = info['lemma']
                # print(f'Lemma hint: {info}')
            elif self._neural_hints == 'pos_feats':
                pos_hint = info['upos']
                feats_hint = info['feats']

            analysis = self._morph.analyze(
                info['text'],
                use_dicts=self._use_dicts,
                lemma_hint=lemma_hint,
                pos_hint=pos_hint,
                feats_hint=feats_hint
            )
            analyses.append(analysis)
        return analyses