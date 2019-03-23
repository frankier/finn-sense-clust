from collections import Counter
import click
from senseclust.groupings import gen_groupings, gen_multi_groupings, inner_join, synset_key_clus
from senseclust.utils import self_xtab
from sklearn.feature_extraction.text import CountVectorizer


def calc_pr(tp, fp, fn):
    # Copypaste from stiff
    if tp == 0:
        return 0, 0, 0
    p = tp / (tp + fp)
    r = tp / (tp + fn)
    return (p, r, 2 * p * r / (p + r))


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

@click.command()
@click.argument("gold", type=click.File('r'))
@click.argument("test", type=click.File('r'))
@click.option('--multi-group/--single-group')
def eval(gold, test, multi_group):
    lemmas = 0
    cnt = Counter()
    line = next(gold)
    assert line.strip() == "pb,wn"
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
    
    print(f"lemmas: {lemmas}")
    print(f"tp: {cnt['tp']}  fp: {cnt['fp']}  fn: {cnt['fn']}  tn: {cnt['tn']}")
    p, r, f1 = calc_pr(cnt['tp'], cnt['fp'], cnt['fn'])
    print(f"p: {p*100:.2f}%  r: {r*100:.2f}%  f1: {f1*100:.2f}%")


if __name__ == "__main__":
    eval()
