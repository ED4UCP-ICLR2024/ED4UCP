from library.tree import Tree, extract_words_from_str_tree

#TODO: averaging should appear in this file as the generate function: MBR-Select vs. MBR-Generate

def select(trees):
    for tree in trees:
        tree.root.set_span(0)
    trees_spans = [set(tree.root.get_all_spans()[1:]) for tree in trees]
    best_tree, best_score = None, -1
    for i, spanset in enumerate(trees_spans):
        score = sum([len(spanset.intersection(other_spanset)) for other_spanset in trees_spans])
        if score > best_score:
            best_tree, best_score = i, score
    return trees[best_tree]