def gen_groupings(inf):
    prev_lemma = None
    cur_map = {}
    for line in inf:
        frame_id, lemma_id = line.strip().split(",", 1)
        lemma, frame_no = frame_id.split(".", 1)
        cur_map.setdefault(frame_no, []).append(lemma_id)
        if prev_lemma is not None and lemma != prev_lemma:
            yield prev_lemma, cur_map
            cur_map = {}
        prev_lemma = lemma
    yield prev_lemma, cur_map


def write_grouping(lemma, cur_map, outf):
    for frame_no, lemma_ids in cur_map.items():
        for lemma_id in lemma_ids:
            outf.write(f"{lemma}.{frame_no},{lemma_id}\n")


def gen_groupings_outer_join(inf1, inf2):
    gen1 = gen_groupings(inf1)
    gen2 = gen_groupings(inf2)
    while 1:
        try:
            lemma1, clus1 = next(gen1)
            lemma2, clus2 = next(gen2)
            while lemma1 != lemma2:
                while lemma1 < lemma2:
                    yield lemma1, clus1, None
                    lemma1, clus1 = next(gen1)
                while lemma2 < lemma1:
                    yield lemma2, None, clus2
                    lemma2, clus2 = next(gen2)
            yield lemma1, clus1, clus2
        except StopIteration:
            break


def gen_groupings_inner_join(inf1, inf2):
    for lemma, clus1, clus2 in gen_groupings_outer_join(inf1, inf2):
        if clus1 is None or clus2 is None:
            continue
        yield lemma, clus1, clus2


def synset_key_clus(clus):
    res = {}
    for clus_idx, synsets in clus.items():
        for synset in synsets:
            res[synset] = clus_idx
    return res


def clus_key_clus(clus):
    res = {}
    for synset, clus_idx in clus.items():
        res.setdefault(clus_idx, []).append(synset)
    return res
