import stanza

from be_hybrid_parser.core import BeHybridParser
from be_hybrid_parser.morph import MorphAnalyzer
from pprint import pprint


parser = BeHybridParser("ён зрабіў гэта падчас", neural_hints=None, use_dicts=True)

pprint(
    parser.analysis,
    sort_dicts=False
)

# morph = MorphAnalyzer()
# morph._ensure_loaded()
#
# pprint(
#     morph._analyze_adp('у'),
#     sort_dicts=False
# )

# print(
#     sorted(
#         [(3, False), (10, True), (9, False), (13, True), (8, False), (3, True)],
#         key=lambda x: (x[1], x[0]),
#         reverse=True
#     )
# )