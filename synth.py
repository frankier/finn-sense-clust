import click
import sys
from networkx.classes.graphviews import subgraph_view
from functools import reduce
from more_itertools import pairwise
import networkx as nx
from networkx.algorithms.clique import find_cliques
from networkx.algorithms.operators.binary import compose
from networkx.algorithms.simple_paths import all_simple_paths
from senseclust.utils.clust import self_xtab

from senseclust.groupings import gen_groupings, write_grouping, grouping_len, filter_grouping_repeats, graph_of_clus, clus_of_graph
from senseclust.utils.cmd import csvin_arg, csvout_arg, wns_arg, predmat_arg, lemma_id_to_synset_id, get_eng_pb_wn_pairs
from senseclust.wordnet import get_lemma_names


@click.group()
def synth():
    pass


def mk_synth(lemma_grouping_pairs, wn):
    all_synth_clus = {}
    for lemma, groupings in lemma_grouping_pairs:
        synth_clus = {}
        for group_num, synsets in groupings.items():
            for synset in synsets:
                for synth_lemma in get_lemma_names(synset, wn):
                    synth_clus.setdefault(synth_lemma, {}).setdefault(group_num, set()).add(synset)
        for lemma, grouping in synth_clus.items():
            filter_grouping_repeats(grouping)
            if grouping_len(grouping) <= 1:
                continue
            all_synth_clus.setdefault(lemma, []).append(grouping)
    return all_synth_clus


def graphs_of_clusts(clusts):
    graph_clusts = []
    for clus in clusts:
        if grouping_len(clus) == 1:
            continue
        graph_clusts.append(graph_of_clus(clus))
    return graph_clusts


def single_contradiction_free_merge(graph_clusts, trace=False, trace_lemma=None):
    # Collect all contradictions
    contradictions = nx.Graph()
    for gc1, gc2 in self_xtab(graph_clusts):
        for e1 in gc1.edges:
            if e1 not in gc2.edges:
                continue
            if gc1.edges[e1]['weight'] == gc2.edges[e1]['weight']:
                continue
            contradictions.add_edge(*e1)
    # Merge graphs
    merged = reduce(compose, graph_clusts)
    if contradictions.edges:
        if trace:
            print(f"Contradiction in clustering for {trace_lemma}", file=sys.stderr)
            print(contradictions.edges, file=sys.stderr)
        merged.remove_edges_from(contradictions.edges)

    return merged


def path_contradiction_free_merge(graph_clusts, trace=False, trace_lemma=None):
    def filter_merge(weight):
        def filtered_graphs():
            for g in graph_clusts:

                def filter_edge(n1, n2):
                    try:
                        return g.edges[n1, n2]['weight'] == weight
                    except KeyError:
                        return False
                yield subgraph_view(g, filter_edge=filter_edge)
        return reduce(compose, filtered_graphs())

    same_merged = filter_merge(1)
    diff_merged = filter_merge(-1)
    rm = set()
    # Iterate through all diff edges
    for u, v in diff_merged.edges:
        # Find all same paths between them
        found_contradiction = False
        for path in all_simple_paths(same_merged, u, v):
            # Found some? Okay delete the negative edge and all positive paths
            found_contradiction = True
            rm.update(pairwise(path))
            if trace:
                print(f"Contradiction in clustering for {trace_lemma}", file=sys.stderr)
                print((u, v), file=sys.stderr)
                print(path, file=sys.stderr)
        if found_contradiction:
            rm.add((u, v))

    # Put them back together and remove all the invalid edges
    merged = compose(same_merged, diff_merged)
    merged.remove_edges_from(rm)
    return merged


def read_conc_clust(csvin):
    conc_clus = {}
    for lemma, groupings in gen_groupings(csvin):
        conc_clus[lemma] = groupings
    return conc_clus


contradictions_arg = \
    click.option('--contradictions', type=click.Choice(["single", "path"]),
                 help='Type of contradictions ')


def proc_write_synth(all_synth_clus, conc_clus, contradictions, csvout):
    for lemma, clusts in sorted(all_synth_clus.items()):
        # Build inc/excl graph
        graph_clusts = graphs_of_clusts(clusts)
        if contradictions == "single":
            merge_func = single_contradiction_free_merge
        else:
            merge_func = path_contradiction_free_merge
        merged = merge_func(graph_clusts, True, lemma)
        # Subtract conc_clus
        if lemma in conc_clus:
            conc_graph = graph_of_clus(conc_clus[lemma])
            conc_edges = conc_graph.edges
            merged.remove_edges_from([e for e in merged.edges if e in conc_edges])
        # Convert back to groupings
        for group_idx, clique in enumerate(find_cliques(merged)):
            grouping = clus_of_graph(merged.subgraph(clique))
            group_lemma = "{}.{:0>2}".format(lemma, group_idx + 1)
            write_grouping(group_lemma, grouping, csvout)


@synth.command()
@csvin_arg
@csvout_arg
@wns_arg
@contradictions_arg
def from_conc(csvin, csvout, wn, contradictions):
    csvout.write("pb,wn\n")
    import stiff.wordnet.fin  # noqa
    next(csvin)
    conc_clus = read_conc_clust(csvin)
    csvin.seek(0)
    next(csvin)
    all_synth_clus = mk_synth(gen_groupings(csvin), wn)
    proc_write_synth(all_synth_clus, conc_clus, contradictions, csvout)


def predmat_frames(pred_matrix):
    result = {}
    for pb, wn in get_eng_pb_wn_pairs(pred_matrix, False):
        lemma, frame_no = pb.rsplit(".", 1)
        result.setdefault(lemma, {}).setdefault(frame_no, []).append(lemma_id_to_synset_id(wn))
    return result.items()


@synth.command()
@csvin_arg
@csvout_arg
@predmat_arg
@wns_arg
@contradictions_arg
def from_predmat(csvin, csvout, pred_matrix, wn, contradictions):
    csvout.write("pb,wn\n")
    import stiff.wordnet.fin  # noqa
    next(csvin)
    conc_clus = read_conc_clust(csvin)
    all_synth_clus = mk_synth(predmat_frames(pred_matrix), wn)
    proc_write_synth(all_synth_clus, conc_clus, contradictions, csvout)


if __name__ == "__main__":
    synth()
