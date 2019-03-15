import click
from groupings import gen_groupings


def calc_pr(tp, fp, fn):
    # Copypaste from stiff
    if tp == 0:
        return 0, 0, 0
    p = tp / (tp + fp)
    r = tp / (tp + fn)
    return (p, r, 2 * p * r / (p + r))


def synset_key_clus(clus):
    res = {}
    for clus_idx, synsets in clus.items():
        for synset in synsets:
            res[synset] = clus_idx
    return res


@click.command()
@click.argument("gold", type=click.File('r'))
@click.argument("test", type=click.File('r'))
def eval(gold, test):
    next(gold)
    gold_gen = gen_groupings(gold)
    test_gen = gen_groupings(test)
    lemmas = 0
    tp = fp = fn = tn = 0
    while 1:
        try:
            gold_lemma, gold_clus = next(gold_gen)
            test_lemma, test_clus = next(test_gen)
            while gold_lemma < test_lemma:
                gold_lemma, gold_clus = next(gold_gen)
            while test_lemma < gold_lemma:
                test_lemma, test_clus = next(test_gen)
        except StopIteration:
            break
        assert gold_lemma == test_lemma
        lemmas += 1
        gold = synset_key_clus(gold_clus)
        test = synset_key_clus(test_clus)
        print(gold_lemma, gold, test)
        pairs = [(gold[gss], test[gss]) for gss in gold if gss in test]
        print("pairs", pairs)
        for idx, (g1, t1) in enumerate(pairs):
            for g2, t2 in pairs[:idx]:
                if g1 == g2:
                    if t1 == t2:
                        tp += 1
                    else:
                        fn += 1
                else:
                    if t1 == t2:
                        fp += 1
                    else:
                        fn += 1
    
    print(f"lemmas: {lemmas}")
    print(f"tp: {fp}; fp: {fp}; fn: {fn}; tn: {tn}")
    p, r, f1 = calc_pr(tp, fp, fn)
    print(f"p: {p}; r: {r}; f1: {f1}")


if __name__ == "__main__":
    eval()
