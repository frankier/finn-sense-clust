from collections import Counter
from senseclust.groupings import gen_groupings, gen_multi_groupings, inner_join, synset_key_clus, outer_join
from senseclust.utils.clust import self_xtab, split_line

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


def tab_one(g1, g2, t1, t2, cnt, incr):
    if g1 == g2:
        if t1 == t2:
            cnt['tp'] += incr
        else:
            cnt['fn'] += incr
    else:
        if t1 == t2:
            cnt['fp'] += incr
        else:
            cnt['tn'] += incr


def eval_clus(gold_clus, test_clus, cnt):
    gold = synset_key_clus(gold_clus)
    test = synset_key_clus(test_clus)
    pairs = []
    for gss in gold:
        if gss not in test:
            raise UnguessedInstanceException(gss)
        pairs.append((gold[gss], test[gss]))
    for (g1, t1), (g2, t2) in self_xtab(pairs):
        tab_one(g1, g2, t1, t2, cnt, 1)


def rand(tp, tn, fp, fn):
    denom = (tp + tn + fp + fn)
    if denom == 0:
        return 0
    return (tp + tn) / denom


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


def pre_cnt_lemmas(gold, test, multi_group):
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


def index_gold_instances(gold, multi_group):
    index_map = {}
    rev_map = []
    idx = 0

    def add(key):
        nonlocal idx
        if key not in index_map:
            index_map[key] = idx
            rev_map.append(key)
            idx += 1
    for line in gold:
        if multi_group:
            frame_id, lemma_id = line.strip().split(",", 1)
            lemma, clus_no, frame_no = frame_id.split(".", 2)
        else:
            lemma, frame_no, lemma_id = split_line(line)
        add((lemma, lemma_id))
    return index_map, rev_map, idx


def add_cluster_partial_cnts(lemma, gc, tc, cnts, index_map):
    gold = synset_key_clus(gc)
    test = synset_key_clus(tc)
    corres = []
    for gss in gold:
        if gss not in test:
            raise UnguessedInstanceException(gss)
        corres.append((gss, gold[gss], test[gss]))
    for ss1, g1, t1 in corres:
        cnt = cnts.setdefault(index_map[(lemma, ss1)], Counter(**ZERO_CONFUSION))
        for ss2, g2, t2 in corres:
            if ss2 == ss1:
                continue
            tab_one(g1, g2, t1, t2, cnt, 0.5)


def pre_cnt_assignments(gold, test, multi_group):
    line = next(gold)
    assert line.strip() in HEADERS
    index_map, rev_map, num_gold_instances = index_gold_instances(gold, multi_group)
    gold.seek(0)
    line = next(gold)
    assert line.strip() in HEADERS
    gold_gen = gen_gold_groupings(gold, multi_group)
    cnts = {}
    for lemma, gold_clus, test_clus in inner_join(gold_gen, gen_groupings(test)):
        if multi_group:
            for gc in gold_clus[0]:
                add_cluster_partial_cnts(lemma, gc, test_clus[0], cnts, index_map)
        else:
            add_cluster_partial_cnts(lemma, gold_clus[0], test_clus[0], cnts, index_map)
    return cnts


def eval_resampled(resample, cnt_map):
    acc = Counter(**ZERO_CONFUSION)
    for sample in resample:
        if sample in cnt_map:
            acc += cnt_map[sample]
    return stats_dict({**ZERO_CONFUSION, **acc})
