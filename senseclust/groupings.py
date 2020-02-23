from .utils.clust import split_line
from more_itertools import peekable


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
    if prev_lemma is not None:
        yield prev_lemma, cur_maps


def gen_groupings(inf):
    prev_lemma = None
    cur_map = {}
    for line in inf:
        lemma, frame_no, lemma_id = split_line(line)
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


def skip_first(csvin, csvout=None):
    csvin = peekable(csvin)
    first_line = csvin.peek().strip()
    if first_line in ("pb,wn", "manann,ref"):
        next(csvin)
        if csvout is not None:
            csvout.write(first_line)
            csvout.write("\n")
    return csvin


def grouping_len(grouping):
    return len([val for li in grouping.values() for val in li])


def filter_grouping_repeats(grouping):
    seen_synsets = set()
    filtered_synsets = set()
    for group_num, synsets in grouping.items():
        for synset in synsets:
            if synset in seen_synsets:
                filtered_synsets.add(synset)
            seen_synsets.add(synset)
    for synsets in grouping.values():
        for filtered_synset in filtered_synsets:
            if filtered_synset in synsets:
                synsets.remove(filtered_synset)
    return grouping, filtered_synsets


def same_diff_of_clus(clus):
    from senseclust.utils.clust import self_xtab
    syns_clus = list(synset_key_clus(clus).items())
    for (s1, c1), (s2, c2) in self_xtab(syns_clus):
        if c1 == c2:
            yield s1, s2, 1
        else:
            yield s1, s2, -1


def graph_of_clus(clus):
    import networkx as nx
    graph = nx.Graph()
    for s1, s2, weight in same_diff_of_clus(clus):
        graph.add_edge(s1, s2, weight=weight)
    return graph


def clus_of_graph(graph):
    partition = {}
    for edge in graph.edges:
        if graph.edges[edge]['weight'] != 1:
            continue
        (l, r) = edge
        if l in partition and r in partition:
            if partition[l] is partition[r]:
                continue
            # Merge sets
            partition[l].update(partition[r])
            for elem in partition[r]:
                partition[elem] = partition[l]
        elif l in partition or r in partition:
            # Add singleton to set
            if r in partition:
                (l, r) = (r, l)
            partition[l].add(r)
            partition[r] = partition[l]
        else:
            # Add new two element set
            partition[l] = partition[r] = {l, r}
    unique_partition = {id(s): s for s in partition.values()}
    return {"{:0>2}".format(idx + 1): v for idx, v in enumerate(unique_partition.values())}
