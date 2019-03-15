import numpy as np
from finntk.emb.autoextend import mk_lemmas_mat, vecs as fiwn_vecs
from finntk.wordnet.reader import fiwn
from fastcluster import linkage
from scipy.cluster import hierarchy

LEMMA_SEP = "-wn-fi-2.0-"

vecs = fiwn_vecs.get_vecs()


def lemmas():
    for entity in vecs.index2entity:
        if LEMMA_SEP not in entity:
            continue
        yield entity


def count(iter):
    return sum((1 for _ in iter))


def clust_all(lim=100):
    all_vecs = vecs[list(lemmas()[:lim])]

    clust = linkage(all_vecs, metric='cosine')


def clust_lemma(lemma_name):
    lemmas = fiwn.lemmas(lemma_name)
    if len(lemmas) == 0:
        return
    lemma_mat = mk_lemmas_mat(lemmas)
    Z = linkage(lemma_mat, metric='cosine')
    return hierarchy.dendrogram(Z, labels=[l.synset().name() for l in lemmas])
