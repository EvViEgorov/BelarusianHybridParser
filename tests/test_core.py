import stanza

from be_hybrid_parser import BeHybridParser
from pprint import pprint


parser = BeHybridParser("на беларускай мове")

pprint(
    parser.analysis,
    sort_dicts=False
)

# pprint(
#     parser.analysis,
#     sort_dicts=False
# )

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