from finntk.emb.autoextend import mk_lemmas_mat, vecs as fiwn_vecs
from finntk.wordnet.reader import fiwn_uniq
from finntk.wordnet.utils import fi2en_post
from fastcluster import linkage
from nltk.corpus import wordnet
from scipy.cluster import hierarchy
from senseclust.utils import graph_clust, group_clust, cos_affinities
from senseclust.exceptions import NoSuchLemmaException
from .base import SenseClusExp
from expcomb.utils import mk_nick

LEMMA_SEP = "-wn-fi-2.0-"

vecs = fiwn_vecs.get_vecs()


def lemmas():
    for entity in vecs.index2entity:
        if LEMMA_SEP not in entity:
            continue
        yield entity


def clust_lemma(lemma_name):
    lemmas = fiwn_uniq.lemmas(lemma_name)
    if len(lemmas) == 0:
        return
    lemma_mat = mk_lemmas_mat(lemmas)
    Z = linkage(lemma_mat, metric='cosine')
    return hierarchy.dendrogram(Z, labels=[l.synset().name() for l in lemmas])


def vec_clust_autoextend_graph(lemma_name, pos):
    lemmas = fiwn_uniq.lemmas(lemma_name, pos)
    if len(lemmas) == 0:
        raise NoSuchLemmaException
    try:
        mat = mk_lemmas_mat(lemmas)
    except KeyError:
        raise NoSuchLemmaException
    else:
        affinities = cos_affinities(mat)
        clust_labels = graph_clust(affinities)
        synsets = [lemma.synset() for lemma in lemmas]
        labels = [fi2en_post(wordnet.ss2of(synset)) for synset in synsets]
        return group_clust(labels, clust_labels)


class Vec(SenseClusExp):
    def __init__(self):
        self.clus_func = vec_clust_autoextend_graph
        super().__init__(
            ("SenseVec",),
            mk_nick("sense-vec"),
            "Sensevec",
            None,
            {},
        )
