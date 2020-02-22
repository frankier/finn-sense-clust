from collections import Counter
import csv
import sys
from finntk.wordnet.utils import fi2en_post
from nltk.corpus import wordnet
import click
import re
from senseclust.groupings import gen_groupings, gen_multi_groupings, write_grouping, clus_key_clus, synset_key_clus, outer_join, skip_first
from senseclust.wordnet import get_lemma_names, WORDNETS
from functools import reduce


GROUPING_INCLUSION_CRITERIA = ('smap-ambg', 'smap2', 'smap', 'ambg', 'none')
PRED_MAT = 'data/PredicateMatrix.v1.3/PredicateMatrix.v1.3.txt'
PROPBANK_DEFNS = 'data/Finnish_PropBank/gen_lemmas/pb-defs.tsv'
MODEL_RE = re.compile(r".*\(tags:[^\)]*model:([^, \)]+)\).*")

csvin_arg = click.argument("csvin", type=click.File('r'))
csvout_arg = click.argument("csvout", type=click.File('w'))
pb_defns_arg = click.option('--pb-defns', default=PROPBANK_DEFNS,
                            help='Path to Finnish PropBank pb-defs.tsv file',
                            type=click.File('r'))
wns_arg = click.option('--wn', type=click.Choice(WORDNETS),
                       default=['fin'], multiple=True,
                       help='Which WordNet (multiple allowed) to use: OMW FiWN, '
                       'FiWN2 or OMW FiWN wikitionary based extensions')


def get_writer(csvout):
    csvout = csv.writer(csvout)
    csvout.writerow(['pb', 'wn'])
    return csvout


def get_reader(propbank):
    return csv.DictReader(propbank, delimiter='\t')


def get_eng_pb_wn_map(matrix, reject_non_english):
    mapping = {}
    matrix = get_reader(matrix)
    for row in matrix:
        if reject_non_english and row['1_ID_LANG'] != 'id:eng':
            continue
        if row['11_WN_SENSE'] != 'wn:NULL':
            pb = row['16_PB_ROLESET'].split(':', 1)[1]
            wn = row['11_WN_SENSE'].split(':', 1)[1]
            mapping.setdefault(pb, set()).add(wn)
    return mapping


@click.group()
def link():
    pass


def include_grouping(filter, wn, lemma, groupings):
    if filter == 'ambg' and len(groupings) <= 1:
        return False
    if filter in ('smap', 'smap2', 'smap-ambg'):
        smap_ambg = set()
        smap_count = 0
        for group_num, synsets in groupings.items():
            for synset in synsets:
                if is_smap(wn, lemma, synset):
                    smap_ambg.add(group_num)
                    smap_count += 1
        if ((filter == 'smap' and smap_count < 1) or
            (filter == 'smap2' and smap_count < 2) or
            (filter == 'smap-ambg' and len(smap_ambg) < 2)):
            return False
    return True


def is_smap(wn, lemma, synset):
    return lemma in get_lemma_names(synset, wn)


@link.command()
@csvin_arg
@click.option('--multi-group/--single-group')
@click.option('--pos')
def get_words(csvin, multi_group, pos):
    import stiff.wordnet.fin
    csvin = skip_first(csvin)
    if multi_group:
        for lemma, multi_groupings in gen_multi_groupings(csvin):
            print(f"{lemma},{pos}")
    else:
        for lemma, groupings in gen_groupings(csvin):
            print(f"{lemma},{pos}")


@link.command()
@csvin_arg
@csvout_arg
@wns_arg
def filter_clus(csvin, csvout, wn):
    import stiff.wordnet.fin
    csvin = skip_first(csvin, csvout)
    for lemma, groupings in gen_groupings(csvin):
        num_synsets = 0
        empty_groups = []
        for group_num, synsets in groupings.items():
            new_synsets = []
            for synset in synsets:
                if not is_smap(wn, lemma, synset):
                    continue
                new_synsets.append(synset)
                num_synsets += 1
            if len(new_synsets):
                groupings[group_num] = new_synsets
            else:
                empty_groups.append(group_num)
        for group_num in empty_groups:
            del groupings[group_num]
        if num_synsets >= 2:
            write_grouping(lemma, groupings, csvout)


def same_diff_of_clus(clus):
    from senseclust.utils import self_xtab
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


def grouping_len(grouping):
    return len([val for li in grouping.values() for val in li])


@link.command()
@csvin_arg
@csvout_arg
@wns_arg
def synth_clus(csvin, csvout, wn):
    csvout.write("pb,wn\n")
    import networkx as nx
    from networkx.algorithms.clique import find_cliques
    from networkx.algorithms.operators.binary import compose
    import stiff.wordnet.fin
    from senseclust.utils import self_xtab
    conc_clus = {}
    next(csvin)
    for lemma, groupings in gen_groupings(csvin):
        conc_clus[lemma] = groupings
    all_synth_clus = {}
    csvin.seek(0)
    next(csvin)
    for lemma, groupings in gen_groupings(csvin):
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
    for lemma, clusts in sorted(all_synth_clus.items()):
        # Build inc/excl graph
        graph_clusts = []
        for clus in clusts:
            if grouping_len(clus) == 1:
                continue
            graph_clusts.append(graph_of_clus(clus))
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
            print(f"Contradiction in clustering for {lemma}", file=sys.stderr)
            print(contradictions.edges, file=sys.stderr)
            merged.remove_edges_from(contradictions.edges)
        # Subtract conc_clus
        if lemma in conc_clus:
            conc_graph = graph_of_clus(conc_clus[lemma])
            merged.remove_edges_from([e for e in merged.edges if e in conc_graph.edges])
        # Convert back to groupings
        for group_idx, clique in enumerate(find_cliques(merged)):
            grouping = clus_of_graph(merged.subgraph(clique))
            group_lemma = "{}.{:0>2}".format(lemma, group_idx + 1)
            write_grouping(group_lemma, grouping, csvout)


@link.command(short_help="Write out lemmas in synset")
@csvin_arg
@wns_arg
def dump(csvin, wn):
    """
    Write out the lemmas in each synset from a CSVIN (frame, synset) relation.
    """
    import stiff.wordnet.fin
    csvin = skip_first(csvin)
    self_mapping = 0
    total = 0
    for line in csvin:
        frame, ssof = line.strip().split(",", 1)
        lemma_names = get_lemma_names(ssof, wn)
        orig_lemma = frame.split(".", 1)[0]
        if orig_lemma in lemma_names:
            self_mapping += 1
        print(frame, " ".join(sorted(lemma_names)))
        total += 1
    print(f"Total: {total}; Self mapping: {self_mapping}; Prop: {self_mapping/total}", file=sys.stderr)


def tri(x):
    return x * (x - 1) // 2


def count_groupings(cnt, has_wiktionary, inc_crit, wn, lemma, groupings):
    all_matchings = sum((
        len(group) for group in groupings.values()
    ))
    cnt['matchings_' + inc_crit] += all_matchings
    cnt['edges_' + inc_crit] += tri(all_matchings)
    if not has_wiktionary:
        smap_matchings = sum((
            1 for synsets in groupings.values()
            for synset in synsets
            if is_smap(wn, lemma, synset)
        ))
        cnt['smap_matchings_' + inc_crit] += smap_matchings
        cnt['smap_edges_' + inc_crit] += tri(smap_matchings)
    same_edges = 0
    diff_edges = 0
    if not has_wiktionary:
        smap_same_edges = 0
        smap_diff_edges = 0
    for s1, s2, weight in same_diff_of_clus(groupings):
        if not has_wiktionary:
            both_smap = is_smap(wn, lemma, s1) and is_smap(wn, lemma, s2)
        if weight == 1:
            same_edges += 1
            if not has_wiktionary and both_smap:
                smap_same_edges += 1
        else:
            diff_edges += 1
            if not has_wiktionary and both_smap:
                smap_diff_edges += 1
    cnt['same_edges_' + inc_crit] += same_edges
    cnt['diff_edges_' + inc_crit] += diff_edges
    if not has_wiktionary:
        cnt['smap_same_edges_' + inc_crit] += smap_same_edges
        cnt['smap_diff_edges_' + inc_crit] += smap_diff_edges


@link.command(short_help="Write out stats for a csv")
@csvin_arg
@wns_arg
@click.option("--multi/--single")
def stats(csvin, wn, multi=False):
    """
    Write out stats for CSVIN
    """
    import stiff.wordnet.fin
    cnt = Counter()
    first_line = next(csvin).strip()
    if first_line == "manann,ref":
        inclusion_criteria = ('ambg', 'none')
        has_wiktionary = True
    elif first_line == "pb,wn":
        inclusion_criteria = GROUPING_INCLUSION_CRITERIA
        has_wiktionary = False
    else:
        assert False

    if multi:
        for lemma, multi_groupings in gen_multi_groupings(csvin):
            for inc_crit in inclusion_criteria:
                if any((include_grouping(inc_crit, wn, lemma, groupings) for groupings in multi_groupings)):
                    cnt['lemmas_' + inc_crit] += 1
                    for groupings in multi_groupings:
                        count_groupings(cnt, has_wiktionary, inc_crit, wn, lemma, groupings)
    else:
        for lemma, groupings in gen_groupings(csvin):
            for inc_crit in inclusion_criteria:
                if include_grouping(inc_crit, wn, lemma, groupings):
                    cnt['lemmas_' + inc_crit] += 1
                    count_groupings(cnt, has_wiktionary, inc_crit, wn, lemma, groupings)

    for k, v in sorted(cnt.items()):
        print(k, v)


@link.command(short_help="Get (frame, synset) rel from FinnPropBank")
@pb_defns_arg
@csvout_arg
def extract_synset_rel(pb_defns, csvout):
    """
    Dumps a (frame, synset) relation to CSVOUT using a the synset_id column from FinnPropBank.
    """
    propbank = get_reader(pb_defns)
    csvout = get_writer(csvout)
    not_found = 0
    found = 0
    for row in propbank:
        pb_finn = "{}.{:0>2}".format(row['base'], row['number'])
        if row['synset_id'] in ("NULL", "666", ""):
            continue
        for raw_synset_id in row['synset_id'].split(","):
            stripped_synset_id = raw_synset_id.strip()
            if not stripped_synset_id:
                continue
            fi_synset_id = stripped_synset_id + "-v"
            try:
                en_synset_id = fi2en_post(fi_synset_id)
            except KeyError:
                print(f"Not found {fi_synset_id} (while processing {pb_finn})", file=sys.stderr)
                not_found += 1
            else:
                csvout.writerow((pb_finn, en_synset_id))
                found += 1
    print(f"Found: {found}; Not found: {not_found}", file=sys.stderr)


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
    return grouping


@link.command(short_help="Remove synsets which are repeated in multiple clusters")
@csvin_arg
@csvout_arg
def filter_repeats(csvin, csvout):
    csvin = skip_first(csvin, csvout)
    for lemma, grouping in gen_groupings(csvin):
        filter_grouping_repeats(grouping)
        write_grouping(lemma, grouping, csvout)


@link.command(short_help="Join FinnPropBank with PredicateMatrix to get (frame, synset) rel")
@click.option('--pred-matrix', default=PRED_MAT,
              help='Path to PredicateMatrix.v?.?.txt TSV file',
              type=click.File('r'))
@pb_defns_arg
@click.option('--reject-non-english/--accept-non-english', default=False,
              help='Accept or reject non-English based mappings')
@click.option('--use-model/--use-link-original', default=True,
              help='Link via the model tag or via the link_original field')
@click.option('--synset/--english-lemmas', default=True,
              help='Map to synset IDs rather than English lemmas')
@csvout_arg
def join_synset(pred_matrix, pb_defns, csvout, reject_non_english, use_model, synset):
    """
    Dumps a (frame, synset) relation to CSVOUT by joining predicate matrix with FinnPropBank.
    """
    # Load mapping from English PropBank senses to English WordNet senses
    mapping = get_eng_pb_wn_map(pred_matrix, reject_non_english)

    # Join with mapping from Finnish to English PropBank
    propbank = get_reader(pb_defns)
    csvout = get_writer(csvout)
    for row in propbank:
        pb_finn = "{}.{:0>2}".format(row['base'], row['number'])
        if use_model:
            match = MODEL_RE.match(row['note'])
            if match:
                pb = match.group(1)
            else:
                pb = None
        else:
            pb = row['link_original']
        if pb == 'none.01':
            pb = None
        if pb is not None and pb in mapping:
            for wn in mapping[pb]:
                if synset:
                    csvout.writerow((pb_finn, wordnet.ss2of(wordnet.lemma_from_key(wn + "::").synset())))
                else:
                    csvout.writerow((pb_finn, wn))


@link.command(short_help="Perform a priority union on (frame, synset) rels")
@click.argument("main_csvin", type=click.File('r'))
@click.argument("secondary_csvin", type=click.File('r'))
@click.argument("csvout", type=click.File('w'))
def priority_union(main_csvin, secondary_csvin, csvout):
    """
    Perform a priority union on (frame, synset) rels. MAIN_CSV gets priority
    """
    disagreements = agreements = 0
    csvout.write("pb,wn\n")
    next(main_csvin)
    next(secondary_csvin)
    for lemma, main_clus, sec_clus in outer_join(gen_groupings(main_csvin), gen_groupings(secondary_csvin)):
        if sec_clus is None:
            write_grouping(lemma, main_clus[0], csvout)
        elif main_clus is None:
            write_grouping(lemma, sec_clus[0], csvout)
        else:
            sk_main_clus = synset_key_clus(main_clus[0])
            sk_sec_clus = synset_key_clus(sec_clus[0])
            for synset, clus_idx in sk_sec_clus.items():
                if synset in sk_main_clus:
                    if sk_main_clus[synset] == clus_idx:
                        agreements += 1
                    else:
                        disagreements += 1
                        print(f"{lemma}: Main says {synset} goes in frame {sk_main_clus[synset]}, but secondary says {clus_idx}", file=sys.stderr)
                else:
                    sk_main_clus[synset] = clus_idx
            write_grouping(lemma, clus_key_clus(sk_main_clus), csvout)
    print(f"Agreements: {agreements}; Disagreements: {disagreements}", file=sys.stderr)


if __name__ == "__main__":
    link()
