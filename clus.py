import sys
import json
import click
import numpy as np
from nltk.corpus import wordnet
from scipy.cluster import hierarchy
from sklearn.cluster import AffinityPropagation
from sklearn import metrics
from senseclust.methods.vec_clust import clust_lemma as vec_clust_lemma
from senseclust.exceptions import NoSuchLemmaException
from senseclust.methods import METHODS, SUPPORTS_WIKTIONARY
from wikiparse.utils.db import get_session


@click.group()
def senseclust():
    pass


CLUS_LEMMAS = ["pitää", "saada", "antaa", "olla", "rakastaa", "kypsyä", "kysyä", "muuttaa"]


def print_clust(clus):
    for lexname, synset_names in clus.items():
        print("# {}".format(lexname))
        for synset_name in synset_names:
            print(synset_name)


def print_lexname_clusters(synsets):
    clus = group_by(((synset.lexname(), synset.name()) for synset in synsets))
    print_clust(clus)


def print_defns(synsets):
    from finntk.wordnet.utils import en2fi_post

    for synset in synsets:
        print(synset.name(), en2fi_post(wordnet.ss2of(synset)), synset.definition())


def print_graph_clust(labels, clust_labels):
    clus = group_clust(labels, clust_labels)
    print_clust(clus)


def dist_clust(dists, labels):
    Z = hierarchy.linkage(dists, method='single')
    minZ = min(Z[:, 2])
    if minZ <= 0.999:
        Z[:, 2] = (Z[:, 2] - minZ) / (1.0 - minZ)
    return hierarchy.dendrogram(Z, labels=labels)


def print_all(lemma_name):
    import matplotlib.pyplot as plt
    langs = get_langs()
    print("# {}".format(lemma_name))
    res = get_sense_sets(lemma_name, langs)
    synsets, lemma_sets = res
    labels = [synset.name() for synset in synsets]

    # Defns
    print_defns(synsets)

    # Lexnames
    print_lexname_clusters(synsets)

    # Get matrix form
    dists, affinities = lemma_measures_of_sets(lemma_sets)

    # Graph cluster
    clust_labels = graph_clust(affinities)
    print_graph_clust(labels, clust_labels)

    # Dist cluster
    plt.figure()
    dist_clust(dists, labels)
    plt.show()

    # Autoextend vec cluster
    plt.figure()
    vec_clust_lemma(lemma_name)
    plt.show()


@senseclust.command("info-demo")
def demo():
    for lemma_name in CLUS_LEMMAS:
        print_all(lemma_name)


@senseclust.command("info")
@click.argument("lemma_name")
def info(lemma_name):
    print_all(lemma_name)


def compare_graph_lex(lemma_name):
    print("# {}".format(lemma_name))
    synsets, lemma_sets = get_sense_sets(lemma_name, get_langs())
    dists, affinities = lemma_measures_of_sets(lemma_sets)
    clust_labels = graph_clust(affinities)
    lexname_labels = [synset.lexname() for synset in synsets]
    cc = lcc(clust_labels, lexname_labels)
    print(lexname_labels, clust_labels, cc)
    synset_labels = [synset.name() for synset in synsets]
    print_graph_clust(synset_labels, lexname_labels)
    print_graph_clust(synset_labels, clust_labels)
    print_graph_clust(synset_labels, cc)
    score = metrics.adjusted_mutual_info_score(lexname_labels, clust_labels)
    print(lemma_name, score)


def lcc(clus1, clus2):
    res = clus1.copy()
    assert len(clus1) == len(clus2)
    n = len(clus1)
    prev_res = res.copy()
    while 1:
        for start_idx, (c1, c2) in enumerate(zip(res, clus2)):
            for idx in range(start_idx + 1, n):
                if clus2[idx] == c2:
                    res[idx] = c1
        if (res == prev_res).all():
            break
        prev_res = res.copy()
    return res


@senseclust.command("compare-all")
def compare_all():
    for lemma_name, synsets, lemma_sets in iter_all():
        compare_graph_lex(lemma_name)


def iter_all():
    for lemma_name in wordnet.all_lemma_names(lang='fin'):
        try:
            synsets, lemma_sets = get_sense_sets(lemma_name, get_langs())
        except NoSuchLemmaException:
            continue
        if len(lemma_sets) == 0:
            continue
        if len(synsets) < 3:
            continue
        yield lemma_name, synsets, lemma_sets


@senseclust.command("compare")
@click.argument("lemma_name")
def compare(lemma_name):
    compare_graph_lex(lemma_name)


@senseclust.command("dump")
def dump():
    for lemma_name, synsets, lemma_sets in iter_all():
        clus = graph_lang_clust(synsets, lemma_sets)
        print(json.dumps([lemma_name, list(clus.values())]))


@senseclust.command("run-graph-lang")
@click.argument("lemmas", type=click.File('r'))
def run_graph_lang(lemmas):
    langs = get_langs()
    for lemma_name in lemmas:
        lemma_name = lemma_name.strip()
        print(repr(lemma_name), file=sys.stderr)
        try:
            synsets, lemma_sets = get_sense_sets(lemma_name, langs)
        except NoSuchLemmaException:
            print(f"No such lemma: {lemma_name}; {lemmas}", file=sys.stderr)
        else:
            synset_map = {synset.name(): synset for synset in synsets}
            clus = graph_lang_clust(synsets, lemma_sets)
            clus_obj = {k: [synset_map[sn] for sn in v] for k, v in clus.items()}
            for k, v in sorted(clus_obj.items()):
                num = k + 1
                for ss in v:
                    off = wordnet.ss2of(ss)
                    print(f"{lemma_name}.{num:02},{off}")


@senseclust.command("run")
@click.argument("method", type=click.Choice(METHODS.keys()))
@click.argument("lemmas", type=click.File('r'))
@click.argument("db", required=False)
def run(method, lemmas, db=None):
    session = None
    for lemma_name in lemmas:
        lemma_name = lemma_name.strip()
        try:
            if method in SUPPORTS_WIKTIONARY:
                if not session:
                    session = get_session(db)
                clus_obj = METHODS[method](lemma_name, include_wiktionary=True, session=session)
            else:
                clus_obj = METHODS[method](lemma_name)
        except NoSuchLemmaException:
            print(f"No such lemma: {lemma_name}", file=sys.stderr)
        else:
            for k, v in sorted(clus_obj.items()):
                num = k + 1
                for ss in v:
                    if method not in SUPPORTS_WIKTIONARY:
                        ss = wordnet.ss2of(ss)
                    print(f"{lemma_name}.{num:02},{ss}")


if __name__ == "__main__":
    senseclust()
