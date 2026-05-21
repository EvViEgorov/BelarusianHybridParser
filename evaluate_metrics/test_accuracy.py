import csv
from idlelib.tree import FileTreeItem
from pathlib import Path
from tqdm import tqdm
from conllu import parse_incr
from be_hybrid_parser.morph import MorphAnalyzer
from collections import defaultdict
import stanza
from stanza.resources.common import ResourcesFileNotFoundError


def str_to_feats(feats_str):
    feats = {}
    if not feats_str or feats_str == '_':
        return feats
    for part in feats_str.split('|'):
        if '=' in part:
            k, v = part.split('=', 1)
            feats[k] = v
    return feats


def feats_to_str(feats_dict):
    if not feats_dict:
        return '_'
    return '|'.join(f"{k}={v}" for k, v in sorted(feats_dict.items()))


def is_match_lemma(pred, gold):
    return pred.get('lemma', '') == gold['lemma']


def is_match_pos(pred, gold):
    return pred.get('POS', '') == gold['pos']


def is_match_feats(pred, gold):
    pred_feats = pred.get('gram_feats', {})
    if isinstance(pred_feats, str):
        pred_feats = str_to_feats(pred_feats)

    gold_feats = gold['feats']
    if not gold_feats:
        return True

    for k, v in gold_feats.items():
        if pred_feats.get(k) != v:
            return False
    return True


def soft_accuracy_for_word(gold_analyses, pred_analyses,
                           check_lemma=True, check_pos=True, check_feats=True):
    if not gold_analyses and not pred_analyses:
        return 1.0
    if not gold_analyses or not pred_analyses:
        return 0.0

    matches = 0
    for pred in pred_analyses:
        feats = pred.get('gram_feats', {})
        if isinstance(feats, str):
            feats = str_to_feats(feats)
            pred_norm = {**pred, 'gram_feats': feats}
        else:
            pred_norm = pred

        for gold in gold_analyses:
            match_result = True
            if check_lemma and not is_match_lemma(pred_norm, gold):
                match_result = False
            if check_pos and not is_match_pos(pred_norm, gold):
                match_result = False
            if check_feats and not is_match_feats(pred_norm, gold):
                match_result = False

            if match_result:
                matches += 1
                break

    denominator = max(len(gold_analyses), len(pred_analyses))
    return matches / denominator


def recall_for_word(gold_analyses, pred_analyses,
                    check_lemma=True, check_pos=True, check_feats=True):
    if not gold_analyses:
        return 1.0
    if not pred_analyses:
        return 0.0

    for pred in pred_analyses:
        feats = pred.get('gram_feats', {})
        if isinstance(feats, str):
            feats = str_to_feats(feats)
            pred_norm = {**pred, 'gram_feats': feats}
        else:
            pred_norm = pred

        for gold in gold_analyses:
            match_result = True
            if check_lemma and not is_match_lemma(pred_norm, gold):
                match_result = False
            if check_pos and not is_match_pos(pred_norm, gold):
                match_result = False
            if check_feats and not is_match_feats(pred_norm, gold):
                match_result = False

            if match_result:
                return 1.0

    return 0.0


def evaluate_parser(conllu_path, max_sentences=None, unknown_only=True, pos_filter=None, use_dicts=True,
                    csv_output=True, hint_modes=None):

    if hint_modes is None:
        hint_modes = [None]

    if pos_filter is None:
        skipped_pos = {'PUNCT', 'SYM', 'X'}
    else:
        skipped_pos = {'PUNCT', 'SYM', 'X'} - set(pos_filter)

    all_overall = {hm: {k: [] for k in ['lemma_only','pos_only','feats_only','pos_feats','lemma_pos_feats']}
                   for hm in hint_modes}
    all_pos = {hm: defaultdict(lambda: {k: [] for k in ['lemma_only','pos_only','feats_only','pos_feats','lemma_pos_feats']})
               for hm in hint_modes}

    all_overall_recall = {hm: {k: [] for k in ['lemma_only','pos_only','feats_only','pos_feats','lemma_pos_feats']}
                          for hm in hint_modes}
    all_pos_recall = {hm: defaultdict(lambda: {k: [] for k in ['lemma_only','pos_only','feats_only','pos_feats','lemma_pos_feats']})
                      for hm in hint_modes}

    all_counts = {hm: defaultdict(int) for hm in hint_modes}
    errors = 0
    total = 0
    parsed = 0

    csv_rows = []

    print("Loading MorphAnalyzer...")
    analyzer = MorphAnalyzer()
    analyzer._ensure_loaded()
    print("Loaded.\n")

    use_stanza = any(m in ('lemma', 'pos_feats') for m in hint_modes)
    if use_stanza:
        print("Loading stanza...")
        try:
            nlp = stanza.Pipeline('be', processors='tokenize,pos,lemma', download_method=None)
        except (ResourcesFileNotFoundError, FileNotFoundError):
            print("Model 'be' not found, downloading...")
            stanza.download('be')
            nlp = stanza.Pipeline('be', processors='tokenize,pos,lemma', download_method=None)
        print("'be' model downloaded.\n")

    with open(conllu_path, 'r', encoding='utf-8') as f:
        all_sentences = list(parse_incr(f))

    total_available = len(all_sentences)
    if max_sentences is not None:
        sentences = all_sentences[:max_sentences]
        print(f"Взято {len(sentences)} из {total_available} предложений\n")
    else:
        sentences = all_sentences
        print(f"Всего предложений: {len(sentences)}\n")

    if pos_filter:
        print(f"Анализируются части речи: {', '.join(pos_filter)}\n")

    for sent_idx, sentence in enumerate(tqdm(sentences, desc="Обработка предложений..."), start=1):
        words = [token['form'] for token in sentence if isinstance(token['id'], int)]
        gold_by_id = {}
        for token in sentence:
            tid = token['id']
            if isinstance(tid, int):
                gold_pos = token['upostag'] or ''
                gold_by_id[tid] = {
                    'lemma': token['lemma'],
                    'pos': gold_pos,
                    'feats': token['feats'] or {}
                }

        stanza_lemmas = [None] * len(words)
        stanza_upos = [None] * len(words)
        stanza_feats = [None] * len(words)

        if use_stanza:
            text = ' '.join(words)
            doc = nlp(text)
            stanza_tokens = []
            for sent in doc.sentences:
                for tok in sent.tokens:
                    stanza_tokens.append(tok)
            if len(stanza_tokens) == len(words):
                for i, tok in enumerate(stanza_tokens):
                    w = tok.words[0]
                    stanza_lemmas[i] = w.lemma
                    stanza_upos[i] = w.upos
                    stanza_feats[i] = w.feats if w.feats else ''

        total += len(words)

        analyses_per_mode = {hm: [] for hm in hint_modes}
        for hm in hint_modes:
            for i, word in enumerate(words):
                lemma_hint = stanza_lemmas[i] if hm == 'lemma' else None
                pos_hint = stanza_upos[i] if hm == 'pos_feats' else None
                feats_hint = stanza_feats[i] if hm == 'pos_feats' else None
                try:
                    analysis = analyzer.analyze(word,
                                                lemma_hint=lemma_hint,
                                                pos_hint=pos_hint,
                                                feats_hint=feats_hint,
                                                use_dicts=use_dicts)
                except Exception as e:
                    errors += 1
                    tqdm.write(f"Error while processing word '{word}': {e}")
                    analysis = []
                analyses_per_mode[hm].append(analysis)

        for i, word in enumerate(words):
            gold_data = gold_by_id.get(i + 1)
            if not gold_data:
                continue

            gold_pos = gold_data['pos']
            if gold_pos in skipped_pos:
                continue
            if pos_filter and gold_pos not in pos_filter:
                continue

            for hm in hint_modes:
                pred_list = analyses_per_mode[hm][i]

                is_unknown = not pred_list or all(
                    pred.get('known_stem', True) == False for pred in pred_list
                )

                if csv_output and is_unknown:
                    gold_feats_str = feats_to_str(gold_data['feats'])
                    if pred_list:
                        for pred in pred_list:
                            pred_feats = pred.get('gram_feats', {})
                            if isinstance(pred_feats, str):
                                pred_feats_str = pred_feats
                                pred_feats = str_to_feats(pred_feats)
                            else:
                                pred_feats_str = feats_to_str(pred_feats)

                            pred_single = [pred]
                            row = {
                                'hint_mode': hm if hm is not None else 'none',
                                'sentence_id': sent_idx,
                                'word': word,
                                'gold_pos': gold_pos,
                                'pred_lemma': pred.get('lemma', ''),
                                'gold_lemma': gold_data['lemma'],
                                'lemma_score': soft_accuracy_for_word(
                                    [gold_data], pred_single,
                                    check_lemma=True, check_pos=False, check_feats=False),
                                'pred_pos': pred.get('POS', ''),
                                'gold_pos_repeat': gold_pos,
                                'pos_score': soft_accuracy_for_word(
                                    [gold_data], pred_single,
                                    check_lemma=False, check_pos=True, check_feats=False),
                                'pred_feats': pred_feats_str,
                                'gold_feats': gold_feats_str,
                                'feats_score': soft_accuracy_for_word(
                                    [gold_data], pred_single,
                                    check_lemma=False, check_pos=False, check_feats=True),
                                'known_stem': pred.get('known_stem', None),
                                'ending_frequency': pred.get('ending_frequency', 0),
                                'all_unknown': is_unknown,
                            }
                            csv_rows.append(row)
                    else:
                        row = {
                            'hint_mode': hm if hm is not None else 'none',
                            'sentence_id': sent_idx,
                            'word': word,
                            'gold_pos': gold_pos,
                            'pred_lemma': '',
                            'gold_lemma': gold_data['lemma'],
                            'lemma_score': 0.0,
                            'pred_pos': '',
                            'gold_pos_repeat': gold_pos,
                            'pos_score': 0.0,
                            'pred_feats': '',
                            'gold_feats': gold_feats_str,
                            'feats_score': 0.0,
                            'known_stem': None,
                            'ending_frequency': 0,
                            'all_unknown': is_unknown,
                        }
                        csv_rows.append(row)

                # Метрики считаем ТОЛЬКО для неизвестных слов
                if unknown_only and not is_unknown:
                    continue

                parsed += 1
                all_counts[hm][gold_pos] += 1

                gold_list = [gold_data]

                # ACCURACY
                sl = soft_accuracy_for_word(gold_list, pred_list,
                                            check_lemma=True, check_pos=False, check_feats=False)
                sp = soft_accuracy_for_word(gold_list, pred_list,
                                            check_lemma=False, check_pos=True, check_feats=False)
                sf = soft_accuracy_for_word(gold_list, pred_list,
                                            check_lemma=False, check_pos=False, check_feats=True)
                spf = soft_accuracy_for_word(gold_list, pred_list,
                                             check_lemma=False, check_pos=True, check_feats=True)
                slpf = soft_accuracy_for_word(gold_list, pred_list,
                                              check_lemma=True, check_pos=True, check_feats=True)

                all_overall[hm]['lemma_only'].append(sl)
                all_overall[hm]['pos_only'].append(sp)
                all_overall[hm]['feats_only'].append(sf)
                all_overall[hm]['pos_feats'].append(spf)
                all_overall[hm]['lemma_pos_feats'].append(slpf)

                pos_s = all_pos[hm][gold_pos]
                pos_s['lemma_only'].append(sl)
                pos_s['pos_only'].append(sp)
                pos_s['feats_only'].append(sf)
                pos_s['pos_feats'].append(spf)
                pos_s['lemma_pos_feats'].append(slpf)

                # RECALL
                rl = recall_for_word(gold_list, pred_list,
                                     check_lemma=True, check_pos=False, check_feats=False)
                rp = recall_for_word(gold_list, pred_list,
                                     check_lemma=False, check_pos=True, check_feats=False)
                rf = recall_for_word(gold_list, pred_list,
                                     check_lemma=False, check_pos=False, check_feats=True)
                rpf = recall_for_word(gold_list, pred_list,
                                      check_lemma=False, check_pos=True, check_feats=True)
                rlpf = recall_for_word(gold_list, pred_list,
                                       check_lemma=True, check_pos=True, check_feats=True)

                all_overall_recall[hm]['lemma_only'].append(rl)
                all_overall_recall[hm]['pos_only'].append(rp)
                all_overall_recall[hm]['feats_only'].append(rf)
                all_overall_recall[hm]['pos_feats'].append(rpf)
                all_overall_recall[hm]['lemma_pos_feats'].append(rlpf)

                pos_r = all_pos_recall[hm][gold_pos]
                pos_r['lemma_only'].append(rl)
                pos_r['pos_only'].append(rp)
                pos_r['feats_only'].append(rf)
                pos_r['pos_feats'].append(rpf)
                pos_r['lemma_pos_feats'].append(rlpf)

    # Собираем результаты
    final_results = {}
    for hm in hint_modes:
        res = {
            'overall': {},
            'overall_recall': {},
            'pos_breakdown': {},
            'pos_breakdown_recall': {},
            'stats': {}
        }
        for metric in ['lemma_only','pos_only','feats_only','pos_feats','lemma_pos_feats']:
            lst = all_overall[hm][metric]
            res['overall'][metric] = sum(lst)/len(lst) if lst else 0.0
            lst_rec = all_overall_recall[hm][metric]
            res['overall_recall'][metric] = sum(lst_rec)/len(lst_rec) if lst_rec else 0.0

        for pos, scores_dict in sorted(all_pos[hm].items()):
            res['pos_breakdown'][pos] = {}
            res['pos_breakdown_recall'][pos] = {}
            for metric in ['lemma_only','pos_only','feats_only','pos_feats','lemma_pos_feats']:
                lst = scores_dict[metric]
                res['pos_breakdown'][pos][metric] = sum(lst)/len(lst) if lst else 0.0
                lst_rec = all_pos_recall[hm][pos][metric]
                res['pos_breakdown_recall'][pos][metric] = sum(lst_rec)/len(lst_rec) if lst_rec else 0.0
            res['pos_breakdown'][pos]['count'] = all_counts[hm][pos]
            res['pos_breakdown_recall'][pos]['count'] = all_counts[hm][pos]

        res['stats'] = {
            'total_words': total,
            'parsed_words': parsed,
            'errors': errors,
            'sentences_processed': len(sentences),
            'pos_filter': pos_filter,
            'max_sentences': max_sentences,
        }
        final_results[str(hm) if hm is not None else 'none'] = res

    if csv_output and csv_rows:
        conllu_path_obj = Path(conllu_path)
        csv_path = conllu_path_obj.parent / f"{conllu_path_obj.stem}_analysis.csv"
        save_csv(csv_rows, csv_path)
        final_results['csv_path'] = str(csv_path)

    return final_results


def save_csv(rows, csv_path):
    if not rows:
        return
    fieldnames = [
        'hint_mode', 'sentence_id', 'word', 'gold_pos',
        'pred_lemma', 'gold_lemma', 'lemma_score',
        'pred_pos', 'gold_pos_repeat', 'pos_score',
        'pred_feats', 'gold_feats', 'feats_score',
        'known_stem', 'ending_frequency', 'all_unknown'
    ]
    with open(csv_path, 'w', encoding='utf-8-sig', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"\nCSV сохранён: {csv_path}")
    print(f"  Строк: {len(rows)}")


def print_results(all_results):
    mode_names = {
        'none': 'БЕЗ ПОДСКАЗОК',
        'lemma': 'ПОДСКАЗКА ЛЕММЫ',
        'pos_feats': 'ПОДСКАЗКА POS+FEATS'
    }
    for mode_key, results in all_results.items():
        if mode_key == 'csv_path':
            continue
        label = mode_names.get(mode_key, mode_key)
        stats = results['stats']
        print("\n" + "=" * 70)
        print(f"  РЕЗУЛЬТАТЫ — {label}")
        print("=" * 70)
        print(f"  Предложений:              {stats['sentences_processed']}")
        if stats['pos_filter']:
            print(f"  Части речи:               {', '.join(stats['pos_filter'])}")
        print(f"  Всего слов оценено:     {stats['parsed_words']}")
        print(f"  Ошибок:                   {stats['errors']}")
        print("-" * 70)

        overall = results['overall']
        overall_recall = results['overall_recall']
        print(f"  {'':<25} {'Accuracy':<10} {'Recall':<10}")
        print(f"  {'Лемма:':<25} {overall['lemma_only']:<10.4f} {overall_recall['lemma_only']:<10.4f}")
        print(f"  {'Часть речи (POS):':<25} {overall['pos_only']:<10.4f} {overall_recall['pos_only']:<10.4f}")
        print(f"  {'Грам. признаки:':<25} {overall['feats_only']:<10.4f} {overall_recall['feats_only']:<10.4f}")
        print(f"  {'POS + признаки:':<25} {overall['pos_feats']:<10.4f} {overall_recall['pos_feats']:<10.4f}")
        print(f"  {'Лемма + POS + призн.:':<25} {overall['lemma_pos_feats']:<10.4f} {overall_recall['lemma_pos_feats']:<10.4f}")

        print("\n  --- По частям речи ---")
        header = f"  {'POS':<8} {'Кол-во':<8} {'Acc-L':<8} {'Rec-L':<8} {'Acc-P':<8} {'Rec-P':<8} {'Acc-F':<8} {'Rec-F':<8}"
        print(header)
        print("  " + "-" * 76)
        for pos, metrics in sorted(results['pos_breakdown'].items()):
            rec_metrics = results['pos_breakdown_recall'][pos]
            print(f"  {pos:<8} {metrics['count']:<8} "
                  f"{metrics['lemma_only']:<8.4f} {rec_metrics['lemma_only']:<8.4f} "
                  f"{metrics['pos_only']:<8.4f} {rec_metrics['pos_only']:<8.4f} "
                  f"{metrics['feats_only']:<8.4f} {rec_metrics['feats_only']:<8.4f}")

    if 'csv_path' in all_results:
        print(f"\n  Подробный CSV: {all_results['csv_path']}")


if __name__ == '__main__':
    conllu_path = 'be_hse-ud-test.conllu'

    MAX_SENTENCES = None  # число-кол-во предложений которые надо проанализировать ЛИБО None
    UNKNOWN_ONLY = True # вести подсчет только по неизвестным основам или по всем словам
    POS_FILTER = None  # список POS-тэгов которые надо анализировать ЛИБО None
    SAVE_CSV = True # Вести лог произведённых анализов в csv-файл
    USE_DICTS = False   # Использовать или не использовать словари для анализа открытых классов (для закрытых словари будут в любом случае)

    results = evaluate_parser(
        conllu_path,
        max_sentences=MAX_SENTENCES,
        unknown_only=UNKNOWN_ONLY,
        pos_filter=POS_FILTER,
        csv_output=SAVE_CSV,
        use_dicts=USE_DICTS,
        hint_modes=[None, 'lemma', 'pos_feats']
    )

    print_results(results)