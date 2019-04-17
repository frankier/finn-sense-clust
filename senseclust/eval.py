from collections import Counter
import click
from senseclust.groupings import gen_groupings, gen_multi_groupings, inner_join, synset_key_clus
from senseclust.utils import self_xtab
from sklearn.feature_extraction.text import CountVectorizer

HEADERS = ("pb,wn", "manann,ref")


def hmean(x, y):
    if x == 0 or y == 0:
        return 0
    return 2 * x * y / (x + y)


def calc_pr(tp, fp, fn):
    # Copypaste from stiff
    if tp == 0:
        return 0, 0, 0
    p = tp / (tp + fp)
    r = tp / (tp + fn)
    return (p, r, hmean(p, r))


def eval_clus(gold_clus, test_clus, cnt):
    gold = synset_key_clus(gold_clus)
    test = synset_key_clus(test_clus)
    pairs = [(gold[gss], test[gss]) for gss in gold if gss in test]
    for (g1, t1), (g2, t2) in self_xtab(pairs):
        if g1 == g2:
            if t1 == t2:
                cnt['tp'] += 1
            else:
                cnt['fn'] += 1
        else:
            if t1 == t2:
                cnt['fp'] += 1
            else:
                cnt['tn'] += 1


def rand(tp, tn, fp, fn):
    denom = (tp + tn + fp + fn)
    if denom == 0:
        return 0
    return tp + tn / denom


def macc(tp, tn, fp, fn):
    denom = (tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)
    if denom == 0:
        return 0
    return (tp * tn - fp * fn) / denom ** 0.5


def eval(gold, test, multi_group):
    lemmas = 0
    cnt = Counter(tp=0, tn=0, fp=0, fn=0)
    line = next(gold)
    assert line.strip() in HEADERS
    if multi_group:
        gold_gen = gen_multi_groupings(gold)
    else:
        gold_gen = gen_groupings(gold)
    for lemma, gold_clus, test_clus in inner_join(gold_gen, gen_groupings(test)):
        lemmas += 1
        if multi_group:
            for gc in gold_clus[0]:
                eval_clus(gc, test_clus[0], cnt)
        else:
            eval_clus(gold_clus[0], test_clus[0], cnt)

    p, r, f1 = calc_pr(cnt['tp'], cnt['fp'], cnt['fn'])
    res = {}
    res["lemmas"] = lemmas
    res["cnt"] = cnt
    tnr = 0 if cnt['tn'] == 0 else cnt['tn'] / (cnt['tn'] + cnt['fp'])
    res["pr"] = {
        "p": p,
        "r": r,
        "f1": f1,
        "tnr": tnr,
    }
    rand_val = rand(**cnt)
    macc_val = macc(**cnt)
    res["o"] = {
        "rand": rand_val,
        "macc": macc_val,
        "hacc": hmean(r, tnr)
    }
    return res
