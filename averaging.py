from library.avg_tree import maximum_shared_spans, span_set_to_tree
from library.tree import Tree, extract_words_from_str_tree, add_words_to_str_tree
from evaluation.compare_trees import compare_trees
from tqdm import tqdm
import numpy as np
import library.MBR as MBR

def right_tree(words):
    string = "(. "+' '.join([f'(. {word})' for word in words])+")"
    return Tree(string)

def _eval(golds, avgs):
    return np.mean(compare_trees(golds, avgs)[1])

def average(reference_trees, ref_str):
    avg = maximum_shared_spans(reference_trees, [1]*len(reference_trees))
    avg = span_set_to_tree(len(reference_trees[0]), avg, ref_str)
    return avg

def median(reference_trees, ref_str):
    solution = MBR.select(reference_trees)
    solution = add_words_to_str_tree(str(solution), extract_words_from_str_tree(ref_str))
    return solution

def aggregate(references, fold='test', right=False, write_to='', aggregate_func=average, source='ptb', mute=False):
    references = [f'{source}/{fold}-{model}.txt' for model in references]
    gold_path = f'{source}/{fold}-gold.txt'

    avgs = []
    cursed_ids = set()
    if mute:
        mytqdm = lambda x, total: x
    else:
        mytqdm = tqdm
    for i, reference_trees in mytqdm(enumerate(zip(*[open(r).readlines() for r in references])), total=len(open(references[0]).readlines())):
        if fold not in ['test', 'susanne']:
            if '\n' in reference_trees:
                cursed_ids.add(i)
                continue
        first_ref_str = reference_trees[0]
        reference_trees = [Tree(r) for r in reference_trees]
        if right:
            reference_trees.append(right_tree(extract_words_from_str_tree(first_ref_str)))
        avg = aggregate_func(reference_trees, first_ref_str)
        avgs.append(avg)

    golds = open(gold_path).readlines()
    if fold!='test':
        golds = [g for i, g in enumerate(golds) if i not in cursed_ids]
    f_score = _eval(golds, avgs)
    if not mute:
        for ref in references:
            print(ref)
        print('cursed_ids:', cursed_ids)
        print('-'*50, len(golds))
        print('Fused F1-Score:', f_score)
        print('-'*50)

    if write_to:
        open(write_to, 'w').write('\n'.join(avgs))

    return f_score