# $ python3 single_eval.py --ref data/ptb-test-gold-filtered.txt --pred OtherProjs/contextual-distortion-parser/src/pred_spans/pred_tree_str.txt

import argparse
from evaluation.compare_trees import compare_trees
import numpy as np
from library.tree import extract_words_from_str_tree, Tree, add_words_to_str_tree

def run(config):
    ref = open(config.ref, 'r').readlines()
    if config.pred.lower()=='oracle':
        pred = [add_words_to_str_tree(str(Tree(line)), extract_words_from_str_tree(line)) for line in ref]
    else:
        pred = open(config.pred, 'r').readlines()
    L = len(ref)
    ref,pred = zip(*[(r, p) for r, p in zip(ref, pred) if p.strip()])
    if L > len(ref):
        print(L-len(ref), 'discarded!!')
    corpus_f1, sent_f1 = compare_trees(ref, pred)
    print(corpus_f1, np.mean(sent_f1))

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--pred', type=str)
    parser.add_argument('--ref', type=str)
    config = parser.parse_args()
    run(config)

if __name__ == '__main__':
    main()