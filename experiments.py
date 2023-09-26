from averaging import aggregate, average, median
import library.naive_heuristics as heu
from evaluation.compare_trees import compare_trees, per_type_compare_trees, per_len_compare_trees
import os
import numpy as np
from itertools import product, combinations
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor
import pandas as pd

ORDER = [
    'on',
    'perturb',
    'npcfg',
    'cpcfg',
    'sdiora',
    'diora',
    'constest',
] # Based on replication avg

# ORDER = [
#     'on',
#     'perturb',
#     'npcfg',
#     'cpcfg',
#     'diora',
#     'sdiora',
#     'constest',
# ] # Based on reported avg

# ORDER = [
#     'on',
#     'npcfg',
#     'perturb',
#     'diora',
#     'sdiora',
#     'cpcfg',
#     'constest',
# ] # Based on first run

def order(run):
    output = []
    for o in ORDER:
        for r in run:
            if r.startswith(o):
                output.append(r)
    if len(output)!=len(run):
        raise Exception("Invalid RUN!")
    return output

RUN1, RUN2, RUN3, RUN4, RUN5 = list(zip(*[
    ['on-seed17',               'on-seed31',                'on',                'on-seed7214',                'on-seed1111',        ],
    ['npcfg-seed3435',          'npcfg',                    'npcfg-seed1234',    'npcfg-seed1313',             'npcfg-seed5555',     ],
    ['cpcfg',                   'cpcfg-seed3435',           'cpcfg-seed1234',    'cpcfg-seed887',              'cpcfg-seed778',      ],
    ['diora',                   'diora-seed35',             'diora-seed74',      'diora-seed1313',             'diora-seed5555',     ],
    ['sdiora-seed1943591871',   'sdiora-seed315',           'sdiora-seed75',     'sdiora-seed1313',            'sdiora-seed442597220'],
    ['constest-best',           'constest-id0',             'constest',          'constest-id1',               'constest-id2',       ],
    ['perturb-layer10',         'perturb-layer12',          'perturb-layer11',   'perturb-layer8',             'perturb-layer9',     ],
]))

RUN1, RUN2, RUN3, RUN4, RUN5 = order(RUN1), order(RUN2), order(RUN3), order(RUN4), order(RUN5)
BEST_COMB = order([
    'on',
    'npcfg',
    'cpcfg',
    'diora-seed1313',
    'sdiora-seed315',
    'constest-best',
    'perturb-layer10',
])
WORST_COMB = order([
    'on-seed31',
    'npcfg-seed3435',
    'cpcfg-seed887',
    'diora',
    'sdiora-seed1313',
    'constest-id0',
    'perturb-layer8',
])
INNERFUSEDS = [f'innerfused-{r.split("-")[0]}' for r in RUN1]
FUSEDS = 'fused-run{}'.format
MBRSel = 'MBRSel-run{}'.format


def produce_rnng_data(path, references=RUN1, aggregate_func=average):
    for fold in ['test','valid','train']:
        aggregate(references, fold=fold, write_to=os.path.join(path, f'{fold}.txt'), aggregate_func=aggregate_func)


def inner_fuse(runs=[RUN1, RUN2, RUN3], write=True, fold='test'):
    for j in range(len(runs[0])):
        write_to = f'ptb-induced-trees/{fold}-innerfused-{runs[0][j].split("-")[0]}.txt' if write else ''
        aggregate([r[j] for r in runs], fold=fold, write_to=write_to)


# def incremental_ensembling_worker(args):
#     run, fold, source = args
#     return aggregate(run, fold=fold, source=source, mute=True)
def incremental_ensembling(runs=[RUN1, RUN2, RUN3, RUN4, RUN5], fold='test', source='ptb'):
    for i in tqdm(range(len(runs[0]))):
        sub_runs = [run[:i+1] for run in runs]
        scores = []
        for run in sub_runs:
        # for run in product(*zip(*sub_runs)):
            scores.append(aggregate(run, fold=fold, source=source, mute=True))
        # with ProcessPoolExecutor() as executor:
        #     args_list = [(run, fold, source) for run in product(*zip(*sub_runs))]
        #     scores = list(executor.map(incremental_ensembling_worker, args_list))
        print('='*30)
        max_, min_ = max(scores), min(scores)
        mean = sum(scores)/len(scores)
        middle = (max_+min_)/2
        print(f'{mean} ; {middle} ± {max_-middle}')
        print('='*30)

def incremental_ensembling_all(runs=[RUN1, RUN2, RUN3, RUN4, RUN5], fold='test', source='ptb'):
    for i in range(len(runs[0])):
        scores = []
        for models in combinations(range(len(runs[0])), i+1):
            this_cobination_scores = []
            sub_runs = [[run[index] for index in models] for run in runs]
            for run in sub_runs:
                this_cobination_scores.append(aggregate(run, fold=fold, source=source))
            scores.append(this_cobination_scores)
        print('='*30)
        for combination in scores:
            max_, min_ = max(combination), min(combination)
            mean_ = sum(combination)/len(combination)
            middle = (max_+min_)/2
            print(f'{mean_} ; {middle} ± {max_-middle}')
            print('='*30)


def naive_heuristics(data='ptb/test-gold.txt'):
    with open(data) as f:
        data = [l.strip() for l in f.readlines() if len(l.strip())]
    print(f"Oracle: {heu.eval(heu.oracle, data)}")
    print(f"Right: {heu.eval(heu.right, data)}")
    print(f"Left: {heu.eval(heu.left, data)}")


def corr(run, fold='test', source='ptb'):
    gold = f'{source}/{fold}-gold.txt'
    trees = [f'{source}/{fold}-{r}.txt' for r in run]
    gold = list(map(str.strip, open(gold).readlines()))
    trees = [list(map(str.strip, open(t).readlines())) for t in trees]
    fscores = [compare_trees(gold, t)[1] for t in trees]
    return np.corrcoef(fscores)

def run_all(runs, aggregate_func=average):
    model_buckets = [[run[i] for run in runs] for i in range(len(runs[0]))]
    for models in tqdm(product(*model_buckets)):
        aggregate(models, aggregate_func=aggregate_func)

def per_tag(run, fold='test', source='ptb'):
    org_gold = open(f'{source}/{fold}-gold.txt').readlines()
    org_trees = [open(f'{source}/{fold}-{r}.txt').readlines() for r in run]
    gold = []
    trees = [[] for i in range(len(org_trees))]
    for g, ts in zip(org_gold, zip(*org_trees)):
        if all(t.strip() for t in ts):
            gold.append(g)
            for i, t in enumerate(ts):
                trees[i].append(t)
    print(len(gold),'/',len(org_gold))
    data = {}
    for t,r in tqdm(zip(trees, run), total=len(run)):
        tag_recalls, tag_coverages = per_type_compare_trees(gold, t)
        data['tag'] = list(tag_recalls.keys())
        data['coverage'] = list(tag_coverages.values())
        data[r.split('-')[0]+'-recall'] = list(tag_recalls.values())
    return pd.DataFrame(data).set_index('tag')

def per_len(run, fold='test', source='ptb'):
    org_gold = open(f'{source}/{fold}-gold.txt').readlines()
    org_trees = [open(f'{source}/{fold}-{r}.txt').readlines() for r in run]
    gold = []
    trees = [[] for i in range(len(org_trees))]
    for g, ts in zip(org_gold, zip(*org_trees)):
        if all(t.strip() for t in ts):
            gold.append(g)
            for i, t in enumerate(ts):
                trees[i].append(t)
    print(len(gold),'/',len(org_gold))
    data = pd.DataFrame()
    for t,r in tqdm(zip(trees, run), total=len(run)):
        len_based_f1, len_coverage = per_len_compare_trees(gold, t)
        for (L, coverage), (L, f1) in zip(len_coverage.items(), len_based_f1.items()):
            data.loc[L, 'coverage'] = coverage
            data.loc[L, r+'-f1-avg'] = np.mean(f1)
            data.loc[L, r+'-f1-std'] = np.std(f1)
    return pd.DataFrame(data)

def not_exists(run, fold, source='ptb'):
    for r in run:
        if not os.path.exists(f'{source}/{fold}-{r}.txt'):
            print(f'{source}/{fold}-{r}.txt')
    print('nothing more')

if __name__ == "__main__":
    # inner_fuse()
    # for i, run in enumerate([RUN1, RUN2, RUN3, RUN4, RUN5]):
    #     aggregate(run, write_to=f'ptb/test-{FUSEDS(i+1)}.txt')
    # aggregate([FUSEDS(i) for i in range(1, 4)])
    # for run in [BEST_COMB, WORST_COMB]:
    #     aggregate(run)
    # aggregate(sum([RUN1, RUN2, RUN3, RUN4, RUN5], []))
    # for i, run in enumerate([RUN1, RUN2, RUN3, RUN4, RUN5]):
    #     aggregate(run, write_to=f'ptb/test-{MBRSel(i+1)}.txt', aggregate_func=median)
    # for run in [BEST_COMB, WORST_COMB]:
    #     aggregate(run, aggregate_func=median)
    # aggregate(sum([RUN1, RUN2, RUN3], []), aggregate_func=median)
    # incremental_ensembling()
    # incremental_ensembling([list(reversed(run)) for run in [RUN1, RUN2, RUN3, RUN4, RUN5]])
    # incremental_ensembling_all()
    # produce_rnng_data('./OtherProjs/urnng/data/fused_worstcomb', WORST_COMB)
    # produce_rnng_data('./OtherProjs/urnng/data/MBRsel_run1', RUN1, aggregate_func=median)
    # naive_heuristics()
    # naive_heuristics('susanne/susanne-gold.txt')
    # aggregate(RUN1, fold='susanne', source='susanne', write_to=f'susanne/susanne-{FUSEDS(1)}.txt')
    # aggregate(RUN1, fold='susanne', source='susanne', aggregate_func=median)
    # incremental_ensembling([list(reversed(run)) for run in [RUN1]], fold='susanne', source='susanne')
    # incremental_ensembling([list((run)) for run in [RUN1]], fold='susanne', source='susanne')
    # print(corr(['sdiora-seed315', 'diora-nli-seed74', 'cpcfg']))
    # print(corr(['cpcfg', 'cpcfg-seed1234', 'cpcfg-seed3435']))
    # run_all([RUN1, RUN2, RUN3])
    # run_all([RUN1, RUN2, RUN3], aggregate_func=median)
    # aggregate(RUN1)
    # aggregate(BEST_COMB, write_to=f'ptb/test-{FUSEDS("Best")}.txt')
    
    for i, run in enumerate([RUN1, RUN2, RUN3, RUN4, RUN5]):
        dfi = per_tag(run+[FUSEDS(i+1)])
        dfi.to_csv(f'per_tag{i+1}.csv')
    
    for i, run in enumerate([RUN1, RUN2, RUN3, RUN4, RUN5]):
        dfi = per_len(run+[FUSEDS(i+1)])
        dfi.to_csv(f'per_len{i+1}.csv')
    
    pass