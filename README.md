# Belarusian Hybrid Parser

A rule-based belarusian language parser with neural model hints feature

## Description

This is a Python library for morphological analysis of belarusian texts.

Parser architecture consists of two layers: rule-based and neural (as a neural layer, stanza 'be' model (based on HSE Belarusian UD Treebank) is used). If neural hints are selected to be used for analysis, they are used as a correction filter for primary analyses made by rule-based layer.

## Installation

The library can be installed using pip package installer:

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

### Analysis format

Text analysis is returned as a list of individual tokens analyses.

An analysis example is shown below:

```
 [{'lemma': 'беларуска',
   'morphemes': ['беларуск', 'ай'],
   'POS': 'NOUN',
   'gram_feats': {'Animacy': 'Anim',
                  'Case': 'Ins',
                  'Gender': 'Fem',
                  'Number': 'Sing'},
   'known_stem': True},
  {'lemma': 'беларускі',
   'morphemes': ['беларуск', 'ай'],
   'POS': 'ADJ',
   'gram_feats': {'Case': 'Gen',
                  'Degree': 'Pos',
                  'Gender': 'Fem',
                  'Number': 'Sing'},
   'known_stem': True},
  {'lemma': 'беларускаць',
   'morphemes': ['беларуска', 'й'],
   'POS': 'VERB',
   'gram_feats': {'Aspect': 'Imp',
                  'Mood': 'Imp',
                  'Number': 'Sing',
                  'Person': '2',
                  'VerbForm': 'Fin',
                  'Voice': 'Act'},
   'known_stem': False}]
```

- known_stem ``bool`` : True for analyses with stems found in stem dicts and False for heuristic analyses