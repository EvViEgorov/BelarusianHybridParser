from be_hybrid_parser.core import BeHybridParser
from be_hybrid_parser.morph import MorphAnalyzer
import pprint

parser = MorphAnalyzer()

pprint.pp(parser._analyze_noun('чалавека'))