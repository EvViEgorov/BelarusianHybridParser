# Belarusian Hybrid Parser

A rule-based belarusian language parser with neural model hints feature

## Installation

```
pip install -i https://test.pypi.org/simple/ BelarusianHybridParser
```

## Usage example

```
from be_hybrid_parser import BeHybridParser

parser = BeHybridParser('Нешта на беларускай мове')
print(parser.analysis)
```

##### Class initialization parameters:

- text ``str`` : A text for analysis

- neural_hints: ``None``, ``'lemma'``, ``'pos_feats'``: Defines which data will be taken from stanza for analysis correction

- use_dicts ``bool``: Defines if stem dicts will be used for NOUN, ADJ, VERB and ADV analysis


##### Class attributes:
- text: Analyzed text 
- tokens: List of tokens
- analysis: Text analysis