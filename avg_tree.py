from library.tree import Tree, Node, add_words_to_str_tree, extract_words_from_str_tree
from collections import defaultdict
from evaluation.compare_trees import compare_trees

def DP(span_scores, length):
    scores = [[0]*length]
    spans = [[[]]*length]
    for level in range(1, length):
        scores.append([])
        spans.append([])
        for begin in range(length-level):
            end = begin+level
            this_span = str((begin, end))
            span_score = span_scores.get(this_span, 0)
            left, best_sub_score = max(([(left, scores[left][begin]+scores[level-left-1][begin+left+1]) for left in range(level)]), key=lambda x: x[1])
            score = span_score + best_sub_score
            new_spans = spans[left][begin] + spans[level-left-1][begin+left+1] + [this_span]
            scores[level].append(score)
            spans[level].append(new_spans)
    return spans[-1][0]

def filter_by_start(spans, start):
    return [span for span in spans if span[0]==start]
def get_largest_span(spans):
    return max(spans, key=lambda span: span[1]-span[0])
def same_span(span1, span2):
    return span1[0]==span2[0] and span1[1]==span2[1]
def filter_includeds(spans, ref):
    return [span for span in spans if (span[0]>=ref[0] and span[1]<=ref[1] and not same_span(span, ref))]
def spans_except(spans, exceptions):
    return [span for span in spans if not any([same_span(span, e) for e in exceptions])]

def build_sub_tree(root_span, spans):
    if root_span[0]==root_span[1]:
        return Node(root_span[0])
    root = Node(root_span)
    left_possibles = filter_by_start(spans, root_span[0])
    if len(left_possibles)==0:
        root.left = Node(root_span[0])
        root.right = build_sub_tree((root_span[0]+1, root_span[1]), spans)
    else:
        left_root_span = get_largest_span(left_possibles)
        left_spans = filter_includeds(spans, left_root_span)
        right_spans = spans_except(spans, left_spans+[left_root_span])
        if left_root_span[1]<root_span[1]:
            root.left = build_sub_tree(left_root_span, left_spans)
            root.right = build_sub_tree((left_root_span[1]+1, root_span[1]), right_spans)
        else:
            root = build_sub_tree(left_root_span, left_spans)
    return root

def sum_span_scores(spans, key_function=str):
    output = defaultdict(lambda: 0)
    for span, score in spans:
        output[key_function(span)] += score
    return output

def forest_to_tree(forest):
    if type(forest) is list:
        return '(. '+' '.join([forest_to_tree(branch) for branch in forest])+')'
    else:
        return f'(. {forest})'

def node2forest(root):
    if root.has_child():
        return [node2forest(root.left), node2forest(root.right)]
    else:
        return root.label

def span_set_to_tree(size, span_set, reference):
    solution = build_sub_tree([0, size-1], span_set)
    solution = forest_to_tree(node2forest(solution))
    solution = add_words_to_str_tree(str(Tree(solution)), extract_words_from_str_tree(reference))
    return solution

def maximum_shared_spans(trees, weights, vote_ignoring_level=0):
    K = len(trees)
    for tree in trees:
        tree.root.set_span(0)
    trees_spans = [tree.root.get_all_spans()[1:] for tree in trees]
    size = len(trees_spans[0])+2
    all_spans = []
    for spans, weight in zip(trees_spans, weights):
        all_spans += [(span, weight) for span in spans]
    span_scores = sum_span_scores(all_spans, key_function=str)
    span_scores = {k: v for k, v in span_scores.items() if v>vote_ignoring_level}
    selected_spans = DP(span_scores, size)
    # selected_spans = [span for span in selected_spans if span_scores.get(span, 0)>0]
    selected_spans = [[int(s) for s in span[1:-1].split(', ')] for span in selected_spans]
    return selected_spans

def main():
    # cpcfg3 = ('(NT-28 (NT-27 (NT-27 (NT-27 (T-47 14) (T-50 million)) (T-51 common)) (T-51 shares)) (NT-4 (T-13 via) (NT-12 (NT-1 (NT-1 (T-37 goldman) (T-9 sachs)) (T-23 &)) (T-17 co))))')
    cpcfg3 = ('(NT-28 (NT-15 (NT-10 (T-55 to) (NT-5 (T-5 a) (T-41 degree))) (T-31 quantum)) (NT-24 (NT-3 (T-7 represents) (NT-19 (NT-20 (T-5 the) (T-40 new)) (T-22 times))) (NT-26 (T-34 that) (NT-13 (T-38 have) (NT-3 (T-14 arrived) (NT-4 (T-13 for) (NT-12 (NT-18 (T-21 producers) (NT-4 (T-13 of) (NT-12 (NT-20 (T-5 the) (T-40 so-called)) (NT-1 (T-18 commodity) (T-25 plastics))))) (NT-26 (T-34 that) (NT-3 (T-19 pervade) (NT-19 (T-60 modern) (T-21 life)))))))))))')
    cpcfg = Tree(cpcfg3)
    npcfg3 = ('(NT-3 (NT-20 (T-8 to) (NT-17 (T-40 a) (T-14 degree))) (NT-3 (T-10 quantum) (NT-13 (NT-26 (T-28 represents) (NT-8 (NT-17 (NT-14 (T-40 the) (T-16 new)) (T-21 times)) (NT-28 (T-51 that) (NT-15 (NT-2 (T-32 have) (T-44 arrived)) (NT-20 (T-52 for) (T-17 producers)))))) (NT-28 (T-23 of) (NT-8 (NT-17 (NT-14 (NT-14 (T-40 the) (T-16 so-called)) (T-53 commodity)) (T-21 plastics)) (NT-28 (T-51 that) (NT-15 (T-46 pervade) (NT-18 (T-11 modern) (T-22 life)))))))))')
    npcfg = Tree(npcfg3)
    on3 = ('(NT-1 (NT-1 (NT-1 (T-1 to) (NT-1 (NT-1 (T-1 a) (NT-1 (T-1 degree) (T-1 quantum))) (NT-1 (T-1 represents) (NT-1 (NT-1 (T-1 the) (T-1 new)) (T-1 times))))) (NT-1 (T-1 that) (NT-1 (T-1 have) (NT-1 (T-1 arrived) (NT-1 (T-1 for) (NT-1 (T-1 producers) (NT-1 (T-1 of) (NT-1 (T-1 the) (NT-1 (T-1 so-called) (NT-1 (T-1 commodity) (T-1 plastics))))))))))) (NT-1 (T-1 that) (NT-1 (T-1 pervade) (NT-1 (T-1 modern) (T-1 life)))))')
    on = Tree(on3)
    prpn3 = ('(NT-1 (T-1 to) (NT-1 (NT-1 (NT-1 (NT-1 (T-1 a) (T-1 degree)) (T-1 quantum)) (NT-1 (T-1 represents) (NT-1 (NT-1 (T-1 the) (NT-1 (T-1 new) (T-1 times))) (NT-1 (T-1 that) (NT-1 (NT-1 (T-1 have) (T-1 arrived)) (NT-1 (T-1 for) (NT-1 (T-1 producers) (NT-1 (T-1 of) (NT-1 (T-1 the) (NT-1 (NT-1 (T-1 so-called) (T-1 commodity)) (T-1 plastics))))))))))) (NT-1 (T-1 that) (NT-1 (T-1 pervade) (NT-1 (T-1 modern) (T-1 life))))))')
    prpn = Tree(prpn3)

    gold3 = '(S (PP (TO To) (NP (DT a) (NN degree))) (NP-SBJ (NNP Quantum)) (VP (VBZ represents) (NP (NP (DT the) (JJ new) (NNS times)) (SBAR (WHNP-1 (WDT that)) (S (VP (VBP have) (VP (VBN arrived))))) (PP (IN for) (NP (NP (NNS producers)) (PP (IN of) (NP (NP (DT the) (JJ so-called) (NN commodity) (NNS plastics)) (SBAR (WHNP-2 (WDT that)) (S (VP (VBP pervade) (NP (JJ modern) (NN life))))))))))))'
    # gold3 = '(NP (NP (QP (CD 14) (CD million)) (JJ common) (NNS shares)) (PP (IN via) (NP (NNP Goldman) (NNP Sachs) (CC &) (NNP Co))))'

    trees = [cpcfg, npcfg, on, prpn]
    # trees = [cpcfg]
    size = len(cpcfg)
    avg = maximum_shared_spans(trees, [1]*len(trees), vote_ignoring_level=0)
    avg = span_set_to_tree(size, avg, cpcfg3)
    print(avg)
    # # avg = '(. (. 0) (. (. (. (. 1) (. 2)) (. (. 3) (. 4))) (. (. 5) (. (. (. (. 6) (. (. 7) (. 8))) (. (. 9) (. (. 10) (. 11)))) (. (. 12) (. (. 13) (. (. 14) (. (. 15) (. 16)))))))))'

    # print(compare_trees([cpcfg3], [avg]))
    # print()

    # # for tree_str in [cpcfg3, npcfg3, on3, prpn3, avg]:
    # for tree_str in [cpcfg3, avg]:
    #     corpus_f1, sent_f1 = compare_trees([gold3], [tree_str])
    #     print(corpus_f1, sent_f1)

if __name__ == "__main__":
    main()