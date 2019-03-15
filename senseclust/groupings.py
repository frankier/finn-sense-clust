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
