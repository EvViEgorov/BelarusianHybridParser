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
        self._adv_suff = None
        self._adv_stems = None
        self._verb_suff = None
        self._verb_stems = None
        self._aux_dict = None
        self._pron_dict = None
        self._num_dict = None
        self._verbpred_dict = None
        self._adp_dict = None
        self._cconj_dict = None
        self._sconj_dict = None
        self._part_dict = None
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
        with open(self._DATA_DIR / 'ADV_suff.json', 'r', encoding='utf-8') as f:
            self._adv_suff = json.load(f)
        with open(self._DATA_DIR / 'ADV_stems.json', 'r', encoding='utf-8') as f:
            self._adv_stems = json.load(f)
        with open(self._DATA_DIR / 'VERB_suff.json', 'r', encoding='utf-8') as f:
            self._verb_suff = json.load(f)
        with open(self._DATA_DIR / 'VERB_stems.json', 'r', encoding='utf-8') as f:
            self._verb_stems = json.load(f)
        with open(self._DATA_DIR / 'AUX_forms.json', 'r', encoding='utf-8') as f:
            self._aux_dict = json.load(f)
        with open(self._DATA_DIR / 'PRON.json', 'r', encoding='utf-8') as f:
            self._pron_dict = json.load(f)
        with open(self._DATA_DIR / 'NUM.json', 'r', encoding='utf-8') as f:
            self._num_dict = json.load(f)
        with open(self._DATA_DIR / 'VERB_PRED.json', 'r', encoding='utf-8') as f:
            self._verbpred_dict = json.load(f)
        with open(self._DATA_DIR / 'ADP.json', 'r', encoding='utf-8') as f:
            self._adp_dict = json.load(f)
        with open(self._DATA_DIR / 'CCONJ.json', 'r', encoding='utf-8') as f:
            self._cconj_dict = json.load(f)
        with open(self._DATA_DIR / 'SCONJ.json', 'r', encoding='utf-8') as f:
            self._sconj_dict = json.load(f)
        with open(self._DATA_DIR / 'PART.json', 'r', encoding='utf-8') as f:
            self._part_dict = json.load(f)
        self._loaded = True

    def _ensure_loaded(self):
        if not self._loaded:
            self._load_resources()

    #
    # ФУНКЦИИ АНАЛИЗА ПО ЧАСТЯМ РЕЧИ
    #

    # ПРИЛАГАТЕЛЬНОЕ
    def _analyze_adj(self, wordform, use_dicts):
        self._ensure_loaded()
        return self._analyze_declinable(
            wordform,
            pos='ADJ',
            stems_dict=self._adj_stems,
            suff_dict=self._adj_suff,
            use_dicts=use_dicts,
            derive_form_feats=self._derive_adj_degree,
            get_possible_lemma_types=self._get_adj_lemma_endings
        )

    def _derive_adj_degree(self, form_stem: str, form_ending: str):
        stem = form_stem.lower()

        if stem.startswith('най'):
            return [{'Degree': 'Sup'}]
        elif stem.endswith(('ейш', 'эйш')):
            return [{'Degree': 'Cmp'}]
        else:
            return [{'Degree': 'Pos'}]

    def _get_adj_lemma_endings(self, stem: str, ending: str):
        possible_by_ending = set([
            (lt, le) for lt in self._adj_suff
            for le, fes in self._adj_suff[lt].items()
            if ending in fes
        ])
        possible_by_stem = set()
        if stem.endswith(('к', 'н', 'х', 'г', 'ц', 'ы')):
            possible_by_stem.add(('і', 'і'))
        if not stem.endswith(('к', 'х', 'г', 'ы')):
            possible_by_stem.add(('ы', 'ы'))
        if stem.endswith(('ын', 'ін', 'аў', 'еў', 'оў', 'ёў')):
            possible_by_stem.add(('', ''))

        result = list(possible_by_ending & possible_by_stem)
        return result

    # СУЩЕСТВИТЕЛЬНОЕ
    def _analyze_noun(self, wordform, use_dicts):
        self._ensure_loaded()
        return self._analyze_declinable(
            wordform,
            pos='NOUN',
            stems_dict=self._noun_stems,
            suff_dict=self._noun_suff,
            use_dicts=use_dicts,
            derive_form_feats=self._derive_feats_noun,
            get_possible_lemma_types=self._get_noun_lemma_endings
        )

    def _derive_feats_noun(self, form_stem: str, form_ending: str) -> dict:
        # выдает Animacy и Gender
        stem = form_stem.lower()

        masc_endings = ['', 'у', 'ам', 'е', 'я', 'ю', 'ем', 'і', 'а', 'ы', 'аў', 'амі',
       'ах', 'ом', 'оў', 'ага', 'аму', 'ым', 'ь', 'яў', 'ям', 'ямі', 'ях',
       'ём', 'ёў', 'ім', 'ай', 'аю', 'ой', 'ою', 'еў', 'ога', 'ому', 'ей',
       'э', 'ьмі', 'яга', 'яму', 'о', 'ёй', 'ёю']
        fem_endings = ['я', 'і', 'ю', 'яй', 'яю', 'а', 'ы', 'у', 'ай', 'аю', '', 'ам',
       'амі', 'ах', 'ь', 'яў', 'ей', 'ям', 'ямі', 'ях', 'е', 'й', 'ая',
       'ую', 'ой', 'ою', 'аў', 'ёй', 'ёю', 'эй', 'э', 'оў', "'ю", 'яя',
       'юю', 'ое', 'ом', 'ьмі', 'ем']
        neut_endings = ['е', 'я', 'ю', 'ем', 'і', 'яў', 'ям', 'ямі', 'ях', 'а', 'у', 'ам',
       'ы', 'аў', 'амі', 'ах', '', 'ае', 'ага', 'аму', 'ым', 'ё', 'ём',
       'ое', 'ога', 'ому', 'о', 'ом', 'оў', 'яе', 'яга', 'яму', 'ім',
       'эй', 'ыма', 'ь', 'ей', 'ьмі', 'ёў', 'ай', 'аю', 'яго']

        poss_animacy = set()
        poss_gender = set()

        if form_ending in masc_endings:
            poss_gender.add('Masc')
        if form_ending in fem_endings:
            poss_gender.add('Fem')
        if form_ending in neut_endings:
            poss_gender.add('Neut')

        poss_animacy.add('Anim')
        poss_animacy.add('Inan')

        result = []

        for anim in poss_animacy:
            for gender in poss_gender:
                result.append({'Animacy': anim, 'Gender': gender})

        return result

    def _get_noun_lemma_endings(self, stem: str, ending: str):
        open_classes = ['0', '1', '2', '3']
        comp_restr_dict = {
            '1': {
                'е': ['з', 'н', "'", 'ц', 'с', 'ў', 'л', 'ь', 'в', 'і'],
                'ь': ['л', 'н', 'з', 'ц', 'с'],
                'ё': ['ц', 'н', "'", 'л', 'ы', 'ў', 'з', 'с'],
                'я': ['н', 'л', 'м', 'б', 'ц', 'з']
            },
            '2': {
                'я': ['і', 'ы', 'н', 'л', 'е', 'э', 'о', 'с', 'ц', 'у', 'а', 'з', "'", 'я', 'ь'],
            },
            '3': {
                'ь': ['ц', 'н', 'л', 'з', 'с'],
                '': ['р', 'ж', 'ч', 'ш', 'б', 'ф', 'ў', 'ц']
            }
        }

        possible_by_ending = set()

        for lt in self._noun_suff:
            for le, fes in self._noun_suff[lt].items():
                if ending in fes and lt in open_classes:
                    restr = comp_restr_dict.get(lt, dict()).get(le, None)
                    if restr:
                        if any(stem.endswith(i) for i in restr):
                            possible_by_ending.add((lt, le))
                    else:
                        possible_by_ending.add((lt, le))

        return possible_by_ending

    # ГЛАГОЛ

    def _analyze_verb(self, wordform, use_dicts):
        self._ensure_loaded()
        return self._analyze_declinable(
            wordform,
            pos='VERB',
            stems_dict=self._verb_stems,
            suff_dict=self._verb_suff,
            use_dicts=use_dicts,
            derive_form_feats=self._derive_feats_verb,
            get_possible_lemma_types=self._get_verb_lemma_endings
        )

    def _derive_feats_verb(self, form_stem: str, form_ending: str) -> list[dict[str, str]]:
        poss_aspect = set()
        poss_voice = set()
        if form_ending.endswith(('ся', 'ца')):
            poss_voice.add('Mid')
        else:
            poss_voice.add('Act')

        poss_aspect.add('Perf')
        poss_aspect.add('Imp')

        result = []

        for aspect in poss_aspect:
            for voice in poss_voice:
                result.append({'Aspect': aspect, 'Voice': voice})

        return result

    def _get_verb_lemma_endings(self, stem: str, ending: str):
        comp_restr_dict = {
            'IA': {
                'ць': ['а', 'я', 'і', 'ы', 'е', 'э', 'о', 'у'],
                'ці': ['с', 'г', 'з', 'б', 'р', 'ў', 'п'],
                'чы': ['г', 'а', 'я', 'к', 'ў']
            },
            'IM': {
                'цца': ['а', 'я', 'і', 'ы', 'о', 'е', 'э', 'у'],
                'ціся': ['с', 'з', 'б', 'р', 'п'],
                'чыся': ['г', 'а', 'я', 'к', 'ў']
            },
            'PA': {
                'ць': ['і', 'а', 'е', 'ы', 'у', 'э', 'я', 'о'],
                'ці': ['с', 'р', 'з', 'б', 'х', 'ў', 'п', 'г'],
                'чы': ['г', 'а', 'я', 'к', 'ў', 'е']
            },
            'PM': {
                'цца': ['і', 'а', 'ы', 'у', 'э', 'е', 'о', 'я'],
                'ціся': ['с', 'р', 'б', 'з', 'п', 'ў'],
                'чыся': ['а', 'г', 'я', 'к', 'е', 'ў']
            }
        }

        possible_by_ending = set()

        for lt in self._verb_suff:
            for le, fes in self._verb_suff[lt].items():
                if ending in fes:
                    restr = comp_restr_dict.get(lt, dict()).get(le, None)
                    if restr:
                        if any(stem.endswith(i) for i in restr):
                            possible_by_ending.add((lt, le))
                    else:
                        possible_by_ending.add((lt, le))

        return possible_by_ending

    # НАРЕЧИЕ
    def _analyze_adv(self, wordform, use_dicts):
        self._ensure_loaded()
        return self._analyze_declinable(
            wordform,
            pos='ADV',
            stems_dict=self._adv_stems,
            suff_dict=self._adv_suff,
            use_dicts=use_dicts,
            derive_form_feats=self._derive_feats_adv,
            # get_possible_lemma_types=self._get_adv_lemma_endings
        )

    def _derive_feats_adv(self, form_stem: str, form_ending: str) -> list[dict[str, str]]:
        stem = form_stem.lower()

        degree = 'Pos'
        if form_ending in ['ей', 'ай', 'эй']:
            degree = 'Cmp'
            if stem.startswith('най'):
                degree = 'Sup'
        result = [{'Degree': degree}]

        return result

    def _get_adv_lemma_endings(self, stem: str, ending: str):
        pass

    # ОБЩАЯ ФУНКЦИЯ АНАЛИЗА
    # Открытые классы
    def _analyze_declinable(
            self,
            wordform: str,
            pos: str,
            stems_dict: dict,
            suff_dict: dict,
            use_dicts: bool,
            derive_form_feats: callable = None,
            get_possible_lemma_types: callable = None
    ):

        wordform = wordform.lower()
        all_analyses = []

        # собираем множество окончаний
        all_endings = set()
        for lemma_type in suff_dict:
            for heuristic_le in suff_dict[lemma_type]:
                for form_ending in suff_dict[lemma_type][heuristic_le]:
                    all_endings.add(form_ending)

        # 1. Поиск в словаре
        if use_dicts:
            for ending in sorted(all_endings, key=len, reverse=True):
                if wordform.endswith(ending):
                    stem_candidate = wordform[:-len(ending)] if ending else wordform

                    # слишком короткие основы в минус
                    if len(stem_candidate) < 2:
                        continue

                    found_in_dict = False

                    for lemma_stem, lemma_types in stems_dict.items():
                        for lemma_type, endings_dict in lemma_types.items():
                            for heuristic_le, stem_info in endings_dict.items():
                                if stem_candidate in stem_info['form_stems']:
                                    lemma = lemma_stem + heuristic_le
                                    const_feats = stem_info.get('const', {})

                                    matching = (
                                        suff_dict.get(lemma_type, {})
                                        .get(heuristic_le, {})
                                        .get(ending, {})
                                    )

                                    for gram_str, freq in matching.items():
                                        variable_feats = self.str_to_feats(gram_str)

                                        if not all(
                                                variable_feats.get(k) == v
                                                for k, v in const_feats.items()
                                        ):
                                            continue

                                        merged_feats = {**variable_feats, **const_feats}

                                        all_analyses.append({
                                            'lemma': lemma,
                                            'morphemes': [stem_candidate, ending],
                                            'pos': pos,
                                            'stem': stem_candidate,
                                            'ending': ending,
                                            'gram_feats': merged_feats,
                                            'frequency': freq,
                                            'known_stem': True
                                        })
                                        found_in_dict = True

                    if found_in_dict:
                        break

        # 2. Эвристика
        if not all_analyses:
            for ending in sorted(all_endings, key=len, reverse=True):
                if wordform.endswith(ending):
                    heuristic_stem = wordform[:-len(ending)] if ending else wordform

                    if len(heuristic_stem) < 2:
                        continue

                    possible_form_feats = derive_form_feats(heuristic_stem, ending) if derive_form_feats else [{}]

                    # Определяем, какие леммные окончания допустимы
                    if get_possible_lemma_types:
                        possible_lemma_ends = get_possible_lemma_types(heuristic_stem, ending)
                    else:
                        # Все lemma_ending, у которых есть данный form_ending
                        possible_lemma_ends = [
                            (lt, le) for lt in suff_dict
                            for le, fes in suff_dict[lt].items()
                            if ending in fes
                        ]

                    for heuristic_lt, heuristic_le in possible_lemma_ends:
                        lemma = heuristic_stem + heuristic_le

                        # Находим все возможные разборы при том же lemma_type и эвристическом lemma ending
                        gram_dict = {}
                        for le, fes in suff_dict[heuristic_lt].items():
                            if heuristic_le == le and ending in fes:
                                gram_dict.update(fes[ending])

                        for gram_str, freq in gram_dict.items():
                            for lex_feats in possible_form_feats:
                                variable_feats = self.str_to_feats(gram_str)

                                if not all(
                                        variable_feats.get(k) == v
                                        for k, v in lex_feats.items()
                                ):
                                    continue

                                merged_feats = {**variable_feats, **lex_feats}

                                all_analyses.append({
                                    'lemma': lemma,
                                    'morphemes': [heuristic_stem, ending],
                                    'pos': pos,
                                    'stem': heuristic_stem,
                                    'ending': ending,
                                    'frequency': freq,
                                    'gram_feats': merged_feats,
                                    'known_stem': False
                                })

                if all_analyses:
                    break

        # 3. Не разобрано
        if not all_analyses:
            return []

        # Сортировка и исключение повторов
        all_analyses.sort(key=lambda x: x['frequency'], reverse=True)

        result = []
        seen = set()
        for analysis in all_analyses:
            sorted_feats = dict(sorted(analysis['gram_feats'].items()))
            key = str((analysis['lemma'], self.feats_to_str(sorted_feats)))
            if key not in seen:
                seen.add(key)
                result.append({
                    'lemma': analysis['lemma'],
                    'morphemes': analysis['morphemes'],
                    'POS': analysis['pos'],
                    'gram_feats': sorted_feats,
                    'known_stem': analysis['known_stem']
                })
        return result

    #
    # НЕИЗМЕНЯЕМЫЕ
    #

    # Функции отдельных частей речи закрытых классов
    # ADP
    def _analyze_adp(self, wordform):
        self._ensure_loaded()
        return self._analyze_indeclinable(
            wordform,
            pos='ADP',
            forms_dict=self._adp_dict
        )

    # CCONJ
    def _analyze_cconj(self, wordform):
        self._ensure_loaded()
        return self._analyze_indeclinable(
            wordform,
            pos='CCONJ',
            forms_dict=self._cconj_dict
        )

    # SCONJ
    def _analyze_sconj(self, wordform):
        self._ensure_loaded()
        return self._analyze_indeclinable(
            wordform,
            pos='SCONJ',
            forms_dict=self._sconj_dict
        )

    # PART
    def _analyze_part(self, wordform):
        self._ensure_loaded()
        return self._analyze_indeclinable(
            wordform,
            pos='PART',
            forms_dict=self._part_dict
        )

    # NUM
    def _analyze_num(self, wordform):
        self._ensure_loaded()
        return self._analyze_indeclinable(
            wordform,
            pos='NUM',
            forms_dict=self._num_dict
        )

    # PRON
    def _analyze_pron(self, wordform):
        self._ensure_loaded()
        return self._analyze_indeclinable(
            wordform,
            pos='PRON',
            forms_dict=self._pron_dict
        )

    # AUX
    def _analyze_aux(self, wordform):
        self._ensure_loaded()
        return self._analyze_indeclinable(
            wordform,
            pos='AUX',
            forms_dict=self._aux_dict
        )

    # PRED
    def _analyze_pred(self, wordform):
        self._ensure_loaded()
        return self._analyze_indeclinable(
            wordform,
            pos='VERB',
            forms_dict=self._verbpred_dict
        )

    # ОБЩАЯ ФУНКЦИЯ АНАЛИЗА
    def _analyze_indeclinable(
            self,
            wordform: str,
            pos: str,
            forms_dict: dict
    ):

        wordform = wordform.lower()
        all_analyses = []

        for lemma, forms in forms_dict.items():
            if not forms and lemma == wordform:
                all_analyses.append({
                    'lemma': lemma,
                    'morphemes': lemma,
                    'pos': pos,
                    'stem': lemma,
                    'ending': '',
                    'gram_feats': '_',
                    'frequency': 100000,
                    'known_stem': True
                })
            else:
                for form_info in forms:
                    for form, gram in form_info.items():
                        if wordform == form:
                            all_analyses.append({
                                'lemma': lemma,
                                'morphemes': lemma,
                                'pos': pos,
                                'stem': lemma,
                                'ending': '',
                                'gram_feats': gram[0],
                                'frequency': 100000,
                                'known_stem': True
                            })

        result = []
        seen = set()
        for analysis in all_analyses:
            feats = analysis['gram_feats']
            key = (analysis['lemma'], feats)
            if key not in seen:
                seen.add(key)
                feats_dict = self.str_to_feats(feats)
                result.append({
                    'lemma': analysis['lemma'],
                    'morphemes': analysis['morphemes'],
                    'POS': analysis['pos'],
                    'gram_feats': feats_dict,
                    'known_stem': analysis['known_stem']
                })
        return result

    def analyze(self, wordform: str, use_dicts,
                lemma_hint: str = None, pos_hint: str = None, feats_hint: str = None):
        self._ensure_loaded()

        results = []

        # Открытые классы
        if pos_hint is None or pos_hint == 'NOUN':
            results.extend(self._analyze_noun(wordform, use_dicts))
        if pos_hint is None or pos_hint == 'ADJ':
            results.extend(self._analyze_adj(wordform, use_dicts))
        if pos_hint is None or pos_hint == 'VERB':
            results.extend(self._analyze_verb(wordform, use_dicts))
            results.extend(self._analyze_pred(wordform))
        if pos_hint is None or pos_hint == 'ADV':
            results.extend(self._analyze_adv(wordform, use_dicts))
        # Закрытые классы
        if pos_hint is None or pos_hint == 'AUX':
            results.extend(self._analyze_aux(wordform))
        if pos_hint is None or pos_hint == 'PRON':
            results.extend(self._analyze_pron(wordform))
        if pos_hint is None or pos_hint == 'NUM':
            results.extend(self._analyze_num(wordform))
        if pos_hint is None or pos_hint == 'ADP':
            results.extend(self._analyze_adp(wordform))
        if pos_hint is None or pos_hint == 'CCONJ':
            results.extend(self._analyze_cconj(wordform))
        if pos_hint is None or pos_hint == 'SCONJ':
            results.extend(self._analyze_sconj(wordform))
        if pos_hint is None or pos_hint == 'PART':
            results.extend(self._analyze_part(wordform))

        # Фильтрация по нейросетевым подсказкам
        if lemma_hint is not None:
            results = [r for r in results if r.get('lemma') == lemma_hint]

        if pos_hint is not None:
            results = [r for r in results if r.get('POS') == pos_hint]

        if feats_hint is not None:
            hint_feats = self.str_to_feats(feats_hint)
            if hint_feats:
                filtered = []
                for r in results:
                    gram = r.get('gram_feats', {})
                    if isinstance(gram, str):
                        gram = self.str_to_feats(gram)
                    if all(gram.get(k) == v for k, v in hint_feats.items()):
                        filtered.append(r)
                results = filtered

        # 3. Сортировка и исключение повторов
        results.sort(key=lambda x: (x.get('known_stem', False), x.get('confidence', 0)), reverse=True)

        seen = set()
        unique_results = []
        for r in results:
            feats = r['gram_feats']
            if isinstance(feats, dict):
                feats_str = self.feats_to_str(feats)
            else:
                feats_str = feats
            key = (r['lemma'], feats_str)
            if key not in seen:
                seen.add(key)
                unique_results.append(r)

        return unique_results

    @staticmethod
    def str_to_feats(feats_str: str):
        feats = {}
        if not feats_str:
            return feats
        for part in feats_str.split('|'):
            if '=' in part:
                k, v = part.split('=', 1)
                feats[k] = v
        result = dict(sorted(feats.items()))
        return result

    @staticmethod
    def feats_to_str(feats_dict: dict):
        return '|'.join(f"{k}={v}" for k, v in sorted(feats_dict.items()))