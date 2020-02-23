from finntk.wordnet.utils import maybe_fi2en_ss, pre_id_to_post
from nltk.corpus import wordnet
from scipy.spatial.distance import pdist, squareform
from senseclust.exceptions import NoSuchLemmaException
from senseclust.utils.clust import graph_clust_grouped, mat_of_sets
from senseclust.wordnet import get_lemma_objs, WORDNETS
from .base import SenseClusExp
from expcomb.utils import mk_nick


def get_langs():
    langs = []

    for lang in wordnet.langs():
        if len(list(wordnet.all_lemma_names(lang=lang))) > 1000:
            langs.append(lang)
    return langs


def lemma_measures_of_sets(lemma_sets):
    mat = mat_of_sets(lemma_sets)
    dists = pdist(mat.todense(), metric='russellrao')
    affinities = squareform((1 - dists))
    return dists, affinities


def graph_lang_clust(labels, lemma_sets, return_centers=False):
    dists, affinities = lemma_measures_of_sets(lemma_sets)
    return graph_clust_grouped(affinities, labels, return_centers)


def get_sense_sets(lemma_name, pos):
    langs = get_langs()
    labels = []
    lemma_sets = []
    id_lemmas = get_lemma_objs(lemma_name, WORDNETS, pos)
    if len(id_lemmas) == 0:
        raise NoSuchLemmaException()
    for synset_id, lemma_objs in id_lemmas.items():
        lemma_set = set()

        def add_lemmas(other_lemmas):
            for lemma in other_lemmas:
                other_lemma_name = lemma.name()
                if other_lemma_name == lemma_name:
                    continue
                lemma_set.add(other_lemma_name)

        def add_omw(synset):
            for lang in langs:
                add_lemmas(synset.lemmas(lang=lang))

        for wn, lemma in lemma_objs:
            synset = lemma.synset()
            if wn == "qf2":
                add_lemmas(synset.lemmas())
                en_synset = maybe_fi2en_ss(synset)
                if en_synset is not None:
                    add_omw(en_synset)
            else:
                add_omw(synset)
        labels.append(pre_id_to_post(synset_id))
        lemma_sets.append(lemma_set)
    return labels, lemma_sets


def label_graph(lemma_name, pos, return_centers=False):
    labels, lemma_sets = get_sense_sets(lemma_name, pos)
    return graph_lang_clust(labels, lemma_sets, return_centers)


class Label(SenseClusExp):
    returns_centers = True

    def __init__(self):
        self.clus_func = label_graph
        super().__init__(
            ("Label",),
            mk_nick("label"),
            "Label",
            None,
            {},
        )
