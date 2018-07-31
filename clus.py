import json
import click
import numpy as np
from nltk.corpus import wordnet
from sklearn.feature_extraction.text import CountVectorizer
from scipy.spatial.distance import pdist, squareform
from scipy.cluster import hierarchy
import matplotlib.pyplot as plt
from sklearn.cluster import AffinityPropagation
from sklearn import metrics
from vec_clust import clust_lemma as vec_clust_lemma


@click.group()
def senseclust():
    pass


def get_langs():
    langs = []

    for lang in wordnet.langs():
        if len(list(wordnet.all_lemma_names(lang=lang))) > 1000:
            langs.append(lang)
    return langs


CLUS_LANG = "fin"
CLUS_LEMMAS = ["pitää", "saada", "antaa", "olla", "rakastaa", "kypsyä", "kysyä", "muuttaa"]


def lemma_mat_of_lemma_sets(lemma_set):
    vectorizer = CountVectorizer(analyzer=lambda x: x)
    return vectorizer.fit_transform(lemma_set)


def graph_of_mat(mat):
    pass


class NoSuchLemmaException(Exception):
    pass


def get_sense_sets(lemma_name, langs):
    synsets = []
    lemma_sets = []
    lemmas = wordnet.lemmas(lemma_name, lang=CLUS_LANG)
    if len(lemmas) == 0:
        raise NoSuchLemmaException()
    for lemma in lemmas:
        synset = lemma.synset()
        lemma_set = set()
        for lang in langs:
            other_lemmas = synset.lemmas(lang=lang)
            for lemma in other_lemmas:
                if lemma.name() == lemma_name:
                    continue
                lemma_set.add(lemma.name())
        synsets.append(synset)
        lemma_sets.append(lemma_set)
    return synsets, lemma_sets


def print_clust(clus):
    for lexname, synset_names in clus.items():
        print("# {}".format(lexname))
        for synset_name in synset_names:
            print(synset_name)


def group_by(it):
    clus = {}
    for k, v in it:
        clus.setdefault(k, []).append(v)
    return clus


def print_lexname_clusters(synsets):
    clus = group_by(((synset.lexname(), synset.name()) for synset in synsets))
    print_clust(clus)


def print_defns(synsets):
    from finntk.wordnet.utils import en2fi_post

    for synset in synsets:
        print(synset.name(), en2fi_post(wordnet.ss2of(synset)), synset.definition())


def group_clust(labels, clust_labels):
    return group_by(((clust_num, labels[idx]) for idx, clust_num in enumerate(clust_labels)))


def print_graph_clust(labels, clust_labels):
    clus = group_clust(labels, clust_labels)
    print_clust(clus)


def lemma_measures_of_sets(lemma_sets):
    mat = lemma_mat_of_lemma_sets(lemma_sets)
    vocab_size = mat.shape[1]
    dists = pdist(mat.todense(), metric='russellrao')
    affinities = squareform((1 - dists) * vocab_size)
    return dists, affinities


def graph_clust(affinities):
    if affinities.shape == (1, 1):
        return [0]
    (n, n) = affinities.shape
    for damping in [0.5, 0.7, 0.9]:
        af = AffinityPropagation(affinity='precomputed', damping=damping).fit(affinities)
        labels = af.labels_
        if not np.any(np.isnan(labels)):
            return labels
    return [0] * n


def dist_clust(dists, labels):
    Z = hierarchy.linkage(dists, method='single')
    minZ = min(Z[:, 2])
    if minZ <= 0.999:
        Z[:, 2] = (Z[:, 2] - minZ) / (1.0 - minZ)
    return hierarchy.dendrogram(Z, labels=labels)


def print_all(lemma_name):
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
        labels = [synset.name() for synset in synsets]
        dists, affinities = lemma_measures_of_sets(lemma_sets)
        clust_labels = graph_clust(affinities)
        clus = group_clust(labels, clust_labels)
        print(json.dumps([lemma_name, list(clus.values())]))


if __name__ == "__main__":
    senseclust()
