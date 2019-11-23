from collections import Counter
from senseclust.groupings import gen_groupings, gen_multi_groupings, inner_join, synset_key_clus, outer_join
from senseclust.utils import self_xtab

HEADERS = ("pb,wn", "manann,ref")
ZERO_CONFUSION = dict(tp=0, tn=0, fp=0, fn=0)


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
    pairs = []
    for gss in gold:
        if gss not in test:
            raise UnguessedInstanceException(gss)
    pairs = [(gold[gss], test[gss])  ]
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


def stats_dict(cnt):
    p, r, f1 = calc_pr(cnt['tp'], cnt['fp'], cnt['fn'])
    res = {}
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


def gen_gold_groupings(gold, multi_group):
    if multi_group:
        return gen_multi_groupings(gold)
    else:
        return gen_groupings(gold)


class UnguessedException(Exception):
    def __init__(self):
        self.gold_fn = None
        self.guess_fn = None

    def __str__(self):
        msg = ""
        if self.gold_fn:
            msg += f" -- found in: {self.gold_fn}"
        if self.guess_fn:
            msg += f" -- missing from: {self.guess_fn}"
        return msg


class UnguessedLemmaException(UnguessedException):
    def __init__(self, missing_lemma):
        self.missing_lemma = missing_lemma
        super().__init__()

    def __str__(self):
        return f"Test does not include missing lemmas {self.missing_lemma}" + super().__str__()


class UnguessedInstanceException(UnguessedException):
    def __init__(self, missing_clus):
        self.missing_clus = missing_clus
        self.missing_lemma = None
        super().__init__()

    def __str__(self):
        return f"Test does not include missing instance {self.missing_lemma}.{self.missing_clus}" + super().__str__()


def eval(gold, test, multi_group):
    lemmas = 0
    cnt = Counter(**ZERO_CONFUSION)
    line = next(gold)
    assert line.strip() in HEADERS
    gold_gen = gen_gold_groupings(gold, multi_group)
    for lemma, gold_clus, test_clus in outer_join(gold_gen, gen_groupings(test)):
        if gold_clus is None:
            # right join
            continue
        if test_clus is None:
            # right join must == inner join
            raise UnguessedLemmaException(lemma)
        lemmas += 1
        try:
            if multi_group:
                for gc in gold_clus[0]:
                    eval_clus(gc, test_clus[0], cnt)
            else:
                eval_clus(gold_clus[0], test_clus[0], cnt)
        except UnguessedInstanceException as exc:
            exc.missing_lemma = lemma
            raise

    res = stats_dict(cnt)
    res["lemmas"] = lemmas
    return res


def pre_cnt_lines(gold, test, multi_group):
    line = next(gold)
    assert line.strip() in HEADERS
    lemma_line_map = {}
    for idx, (lemma, groups) in enumerate(gen_gold_groupings(gold, multi_group)):
        lemma_line_map[lemma] = idx
    cnts = {}
    gold.seek(0)
    line = next(gold)
    assert line.strip() in HEADERS
    gold_gen = gen_gold_groupings(gold, multi_group)
    for lemma, gold_clus, test_clus in inner_join(gold_gen, gen_groupings(test)):
        cnt = Counter(**ZERO_CONFUSION)
        if multi_group:
            for gc in gold_clus[0]:
                eval_clus(gc, test_clus[0], cnt)
        else:
            eval_clus(gold_clus[0], test_clus[0], cnt)
        cnts[lemma_line_map[lemma]] = cnt
    return cnts


def eval_resampled(resample, cnt_map):
    acc = Counter(**ZERO_CONFUSION)
    for sample in resample:
        if sample in cnt_map:
            acc += cnt_map[sample]
    return stats_dict({**ZERO_CONFUSION, **acc})
