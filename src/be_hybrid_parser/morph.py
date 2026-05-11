import json
from collections import defaultdict
from pathlib import Path


class MorphAnalyzer:

    _MODULE_DIR = Path(__file__).parent
    _DATA_DIR = _MODULE_DIR / 'data'

    def __init__(self):
        self._adj_suff = None
        self._adj_stems = None
        self._noun_suff = None
        self._noun_stems = None
        self._loaded = False

    # загрузка словарей
    def _load_resources(self):
        with open(self._DATA_DIR / 'ADJ_suff.json', 'r', encoding='utf-8') as f:
            self._adj_suff = json.load(f)
        with open(self._DATA_DIR / 'ADJ_stems.json', 'r', encoding='utf-8') as f:
            self._adj_stems = json.load(f)
        with open(self._DATA_DIR / 'NOUN_suff.json', 'r', encoding='utf-8') as f:
            self._noun_suff = json.load(f)
        with open(self._DATA_DIR / 'NOUN_stems.json', 'r', encoding='utf-8') as f:
            self._noun_stems = json.load(f)
        self._loaded = True

    def _ensure_loaded(self):
        if not self._loaded:
            self._load_resources()

    # ФУНКЦИИ АНАЛИЗА ПО ЧАСТЯМ РЕЧИ
    def _analyze_adjective(self, wordform):
        self._ensure_loaded()
        return self._analyze_word(
            wordform,
            stems_dict = self._adj_stems,
            suff_dict = self._adj_suff
        )

    def _analyze_noun(self, wordform):
        self._ensure_loaded()
        return self._analyze_word(
            wordform,
            stems_dict = self._noun_stems,
            suff_dict = self._noun_suff
        )

    # ОБЩАЯ ФУНКЦИЯ АНАЛИЗА
    def _analyze_word(
            self,
            wordform: str,
            stems_dict: dict,
            suff_dict: dict
    ):

        wordform = wordform.lower()
        all_analyses = []  # список анализов вида (ending, stem, tags_str, freq)

        # 1. Пробуем поиск в словаре. Смотрим окончания от самых длинных до самых коротких
        for ending in sorted(suff_dict.keys(), key=len, reverse=True):
            if wordform.endswith(ending):
                stem_candidate = wordform[:-len(ending)] if ending else wordform

                # однобуквенные основы в минус
                if len(stem_candidate) < 2:
                    continue

                for lemma_stem, stem_info in stems_dict.items():
                    for word_type, type_info in stem_info['types'].items():
                        for allomorph in type_info['allomorphs']:
                            if stem_candidate == allomorph:
                                # Нашли совпадение через алломорф
                                lemma = lemma_stem + word_type
                                constant_feats = type_info.get('constant_feats', {})

                                matching_suffixes = suff_dict[ending].get(word_type, {})

                                for tags_str, freq in matching_suffixes.items():
                                    # мёрджим постоянные и вычисленные признаки
                                    variable_feats = self.parse_ud_feats(tags_str)
                                    merged_feats = {**variable_feats, **constant_feats}
                                    merged_str = feats_to_str(merged_feats)
                                    all_analyses.append({
                                        'lemma': lemma,
                                        'stem': allomorph,
                                        'ending': ending,
                                        'gram_feats': merged_feats,
                                        'frequency': freq
                                    })

                if all_analyses:
                    break  # берём только самое длинное подходящее окончание

        # 2. Если ничего не нашли - будет эвристика
        if not all_analyses:
            # еще раз пройдемся по окончаниям
            for ending in sorted(suff_dict.keys(), key=len, reverse=True):
                if wordform.endswith(ending):
                    heuristic_stem = wordform[:-len(ending)] if ending else wordform

                    # однобуквенные основы в минус
                    if len(heuristic_stem) < 2:
                        continue

                    possible_lemma_endings = list(suff_dict[ending].keys())

                    for poss_lemma_end in possible_lemma_endings:
                        lemma = heuristic_stem + poss_lemma_end

                        matching_suffixes = suff_dict[ending].get(poss_lemma_end, {})

                        if not matching_suffixes:
                            # Если нет точного совпадения по lemma_ending,
                            # берём все разборы для данного form_ending
                            for lemma_end, gram_dict in suff_dict[ending].items():
                                for tags_str, freq in gram_dict.items():
                                    all_analyses.append({
                                        'lemma': lemma,
                                        'stem': heuristic_stem,
                                        'ending': ending,
                                        'gram_feats': tags_str,
                                        'frequency': freq
                                    })
                        else:
                            for tags_str, freq in matching_suffixes.items():
                                all_analyses.append({
                                    'lemma': lemma,
                                    'stem': heuristic_stem,
                                    'ending': ending,
                                    'gram_feats': tags_str,
                                    'frequency': freq
                                })

                if all_analyses:
                    break  # берём только самое длинное подходящее окончание

        # 3. Если все равно не парсится - констатируем невозможность разобрать
        if not all_analyses:
            return []

        # Сортируем по убыванию частоты
        all_analyses.sort(key=lambda x: x['frequency'], reverse=True)

        # Формируем результат
        result = []
        seen = set()
        for analysis in all_analyses:
            key = str((analysis['lemma'], analysis['gram_feats']))
            if key not in seen:
                seen.add(key)
                result.append({
                    'lemma': analysis['lemma'],
                    'gram_feats': analysis['gram_feats'],
                    'ending_frequency': analysis['frequency']
                })
        return result

    @staticmethod
    def parse_ud_feats(feats_str: str):
        feats = {}
        if not feats_str:
            return feats
        for part in feats_str.split('|'):
            if '=' in part:
                k, v = part.split('=', 1)
                feats[k] = v
        return feats

    @staticmethod
    def feats_to_str(feats_dict: dict):
        return '|'.join(f"{k}={v}" for k, v in sorted(feats_dict.items()))



_adj_suff = None
_adj_stems = None
_noun_suff = None
_noun_stems = None
_noun_suff_match = None


def _ensure_loaded():
    global _adj_suff, _adj_stems, _noun_suff, _noun_stems, _noun_suff_match
    if None in (_adj_suff, _adj_stems, _noun_suff, _noun_stems, _noun_suff_match):
        _adj_suff, _adj_stems, _noun_suff, _noun_stems, _noun_suff_match = load_resources()


# UD в словарь и назад
def parse_ud_feats(feats_str):
    feats = {}
    if not feats_str:
        return feats
    for part in feats_str.split('|'):
        if '=' in part:
            k, v = part.split('=', 1)
            feats[k] = v
    return feats

def feats_to_str(feats_dict):
    return '|'.join(f"{k}={v}" for k, v in sorted(feats_dict.items()))


# БЛОК ФУНКЦИЙ АНАЛИЗА ЧАСТЕЙ РЕЧИ
def _analyze_adjective(wordform):
    def _get_heuristic_lemma_type(lemma_stem, ending):
        if ending:
            if lemma_stem.endswith(('к', 'х', 'г', 'ы')):
                return 'і'
            else:
                return 'ы'
        else:
            if lemma_stem.endswith(('ын', 'ін', 'аў', 'еў', 'оў', 'ёў', 'яў')):
                return ''
            else:
                return None

    _ensure_loaded()

    wordform = wordform.lower()
    all_analyses = []  # список анализов вида (ending, stem, tags_str, freq)

    # Поиск в словаре от длинных до коротких окончаний
    for ending in sorted(_adj_suff.keys(), key=len, reverse=True):
        if wordform.endswith(ending):
            stem_candidate = wordform[:-len(ending)] if ending else wordform

            # однобуквенные основы не разбираем
            if len(stem_candidate) < 2:
                continue

            for lemma_stem, lemma_info in _adj_stems.items():
                for allomorph in lemma_info['allomorphs']:
                    if stem_candidate == allomorph:
                        # Нашли совпадение через алломорф
                        lemma_ending = lemma_info['type']
                        lemma = lemma_stem + lemma_ending

                        for tags_str, freq in _adj_suff[ending].items():
                            all_analyses.append({
                                'lemma': lemma,
                                'stem': allomorph,
                                'ending': ending,
                                'gram_feats': tags_str,
                                'frequency': freq
                            })

            if all_analyses:
                break  # берём только самое длинное подходящее окончание

    # 2. Если ничего не нашли - будет эвристика
    if not all_analyses:
        # еще раз пройдемся по окончаниям
        for ending in sorted(_adj_suff.keys(), key=len, reverse=True):
            if wordform.endswith(ending):
                heuristic_stem = wordform[:-len(ending)] if ending else wordform

                # однобуквенные основы в минус
                if len(heuristic_stem) < 2:
                    continue

                lemma_ending = _get_heuristic_lemma_type(heuristic_stem, ending)
                if lemma_ending is None:
                    break
                lemma = heuristic_stem + lemma_ending
                if not lemma.endswith(('ы', 'і', 'ын', 'ін', 'аў', 'еў', 'оў', 'ёў', 'яў')):
                    break

                # Добавляем все возможные грам. теги для этого окончания
                for tags_str, freq in _adj_suff[ending].items():
                    all_analyses.append({
                        'lemma': lemma,
                        'stem': heuristic_stem,
                        'ending': ending,
                        'gram_feats': tags_str,
                        'frequency': freq
                    })

            if all_analyses:
                break  # берём только самое длинное подходящее окончание

    # 3. Если все равно не парсится - констатируем невозможность разобрать
    if not all_analyses:
        return []

    # Сортируем по убыванию частоты
    all_analyses.sort(key=lambda x: x['frequency'], reverse=True)

    # Формируем результат
    result = []
    seen = set()
    for analysis in all_analyses:
        key = (analysis['lemma'], analysis['gram_feats'])
        if key not in seen:
            seen.add(key)
            result.append({
                'lemma': analysis['lemma'],
                'gram_feats': analysis['gram_feats'],
                'ending_frequency': analysis['frequency']
            })
    return result


def analyze_noun(wordform):
    def _get_heuristic_lemma_types(lemma_stem, ending):
        possible_endings = set()
        for lemma_ending in _noun_suff_match:
            for form_ending in _noun_suff_match[lemma_ending]:
                if ending == form_ending:
                    possible_endings.add(lemma_ending)

        # проверяем на -р
        soft = {'я', 'ё', 'ю', 'і', 'е', 'ь'}
        if lemma_stem.endswith('р'):
            possible_endings = {i for i in possible_endings if i not in soft}

        possible_endings = list(possible_endings)
        if possible_endings:
            return possible_endings
        else:
            return ['']

    _ensure_loaded()

    wordform = wordform.lower()
    all_analyses = []  # список анализов вида (ending, stem, tags_str, freq)

    # 1. Пробуем поиск в словаре. Смотрим окончания от самых длинных до самых коротких
    for ending in sorted(_noun_suff.keys(), key=len, reverse=True):
        if wordform.endswith(ending):
            stem_candidate = wordform[:-len(ending)] if ending else wordform

            # однобуквенные основы в минус
            if len(stem_candidate) < 2:
                continue

            for lemma_stem in _noun_stems:
                for noun_type in _noun_stems[lemma_stem]:
                    for allomorph in _noun_stems[lemma_stem][noun_type]:
                        if stem_candidate == allomorph:
                            # Нашли совпадение через алломорф
                            lemma = lemma_stem + noun_type

                            for tags_str, freq in _noun_suff[ending].items():
                                all_analyses.append({
                                    'lemma': lemma,
                                    'stem': allomorph,
                                    'ending': ending,
                                    'gram_feats': tags_str,
                                    'frequency': freq
                                })

            if all_analyses:
                break  # берём только самое длинное подходящее окончание

    # 2. Если ничего не нашли - будет эвристика
    if not all_analyses:
        # еще раз пройдемся по окончаниям
        for ending in sorted(_noun_suff.keys(), key=len, reverse=True):
            if wordform.endswith(ending):
                heuristic_stem = wordform[:-len(ending)] if ending else wordform

                # однобуквенные основы в минус
                if len(heuristic_stem) < 2:
                    continue

                possible_lemma_endings = _get_heuristic_lemma_types(heuristic_stem, ending)
                for poss_lemma_end in possible_lemma_endings:
                    lemma = heuristic_stem + poss_lemma_end

                    # Добавляем все возможные грам. теги для этого окончания
                    for tags_str, freq in _noun_suff[ending].items():
                        all_analyses.append({
                            'lemma': lemma,
                            'stem': heuristic_stem,
                            'ending': ending,
                            'gram_feats': tags_str,
                            'frequency': freq
                        })

            if all_analyses:
                break  # берём только самое длинное подходящее окончание

    # 3. Если все равно не парсится - констатируем невозможность разобрать
    if not all_analyses:
        return []

    # Сортируем по убыванию частоты
    all_analyses.sort(key=lambda x: x['frequency'], reverse=True)

    # Формируем результат
    result = []
    seen = set()
    for analysis in all_analyses:
        key = (analysis['lemma'], analysis['gram_feats'])
        if key not in seen:
            seen.add(key)
            result.append({
                'lemma': analysis['lemma'],
                'gram_feats': analysis['gram_feats'],
                'ending_frequency': analysis['frequency']
            })
    return result