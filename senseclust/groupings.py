def gen_multi_groupings(inf):
    prev_lemma = None
    prev_clus_no = None
    cur_maps = [{}]
    for line in inf:
        frame_id, lemma_id = line.strip().split(",", 1)
        lemma, clus_no, frame_no = frame_id.split(".", 2)
        if prev_lemma is not None and lemma != prev_lemma:
            yield prev_lemma, cur_maps
            cur_maps = [{}]
        elif prev_clus_no is not None and clus_no != prev_clus_no:
            cur_maps.append({})
        cur_maps[-1].setdefault(frame_no, []).append(lemma_id)
        prev_lemma = lemma
        prev_clus_no = clus_no
    yield prev_lemma, cur_maps


def gen_groupings(inf):
    prev_lemma = None
    cur_map = {}
    for line in inf:
        frame_id, lemma_id = line.strip().split(",", 1)
        lemma, frame_no = frame_id.split(".", 1)
        if prev_lemma is not None and lemma != prev_lemma:
            yield prev_lemma, cur_map
            cur_map = {}
        cur_map.setdefault(frame_no, []).append(lemma_id)
        prev_lemma = lemma
    if prev_lemma is not None:
        yield prev_lemma, cur_map


def write_grouping(lemma, cur_map, outf):
    for frame_no, lemma_ids in cur_map.items():
        for lemma_id in lemma_ids:
            outf.write(f"{lemma}.{frame_no},{lemma_id}\n")


def outer_join(gen1, gen2):
    lemma1 = clus1 = lemma2 = clus2 = None
    while 1:
        try:
            def next1():
                nonlocal lemma1, clus1
                item1 = next(gen1)
                lemma1, clus1 = item1[0], item1[1:]
            def next2():
                nonlocal lemma2, clus2
                item2 = next(gen2)
                lemma2, clus2 = item2[0], item2[1:]
            next1()
            next2()
            while lemma1 != lemma2:
                while lemma1 < lemma2:
                    yield lemma1, clus1, None
                    next1()
                while lemma2 < lemma1:
                    yield lemma2, None, clus2
                    next2()
            yield lemma1, clus1, clus2
        except StopIteration:
            break


def inner_join(gen1, gen2):
    for lemma, clus1, clus2 in outer_join(gen1, gen2):
        if clus1 is None or clus2 is None:
            continue
        yield lemma, clus1, clus2


def synset_key_clus(clus):
    res = {}
    for clus_idx, synsets in clus.items():
        for synset in synsets:
            assert synset not in res
            res[synset] = clus_idx
    return res


def clus_key_clus(clus):
    res = {}
    for synset, clus_idx in clus.items():
        res.setdefault(clus_idx, []).append(synset)
    return res
