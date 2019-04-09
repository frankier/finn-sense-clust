from nltk.corpus import wordnet
from scipy.spatial.distance import pdist, squareform
from senseclust.consts import CLUS_LANG
from senseclust.exceptions import NoSuchLemmaException
from senseclust.utils import graph_clust, mat_of_sets, group_clust


def get_langs():
    langs = []

    for lang in wordnet.langs():
        if len(list(wordnet.all_lemma_names(lang=lang))) > 1000:
            langs.append(lang)
    return langs


def lemma_measures_of_sets(lemma_sets):
    mat = mat_of_sets(lemma_sets)
    vocab_size = mat.shape[1]
    dists = pdist(mat.todense(), metric='russellrao')
    affinities = squareform((1 - dists) * vocab_size)
    return dists, affinities


def graph_lang_clust(synsets, lemma_sets):
    labels = [wordnet.ss2of(synset) for synset in synsets]
    dists, affinities = lemma_measures_of_sets(lemma_sets)
    clust_labels = graph_clust(affinities)
    return group_clust(labels, clust_labels)


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


def label_graph(lemma_name):
    langs = get_langs()
    synsets, lemma_sets = get_sense_sets(lemma_name, langs)
    return graph_lang_clust(synsets, lemma_sets)
