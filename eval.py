import click
from senseclust.groupings import gen_groupings_inner_join, synset_key_clus


def calc_pr(tp, fp, fn):
    # Copypaste from stiff
    if tp == 0:
        return 0, 0, 0
    p = tp / (tp + fp)
    r = tp / (tp + fn)
    return (p, r, 2 * p * r / (p + r))


@click.command()
@click.argument("gold", type=click.File('r'))
@click.argument("test", type=click.File('r'))
def eval(gold, test):
    lemmas = 0
    tp = fp = fn = tn = 0
    next(gold)
    for lemma, gold_clus, test_clus in gen_groupings_inner_join(gold, test):
        lemmas += 1
        gold = synset_key_clus(gold_clus)
        test = synset_key_clus(test_clus)
        pairs = [(gold[gss], test[gss]) for gss in gold if gss in test]
        if len(pairs) >= 2:
            print("# " + lemma)
            print(gold)
            print(test)
            print(pairs)
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
    print(f"tp: {tp}; fp: {fp}; fn: {fn}; tn: {tn}")
    p, r, f1 = calc_pr(tp, fp, fn)
    print(f"p: {p}; r: {r}; f1: {f1}")


if __name__ == "__main__":
    eval()
