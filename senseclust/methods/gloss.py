from gensim.corpora import Dictionary
from gensim.similarities import SoftCosineSimilarity
from finntk.emb.fasttext import vecs
import numpy as np
from senseclust.utils import graph_clust, group_clust, get_defns, unclusterable_default
from .base import SenseClusExp
from expcomb.utils import mk_nick
from wikiparse.utils.db import get_session


def softcos(defns, return_centers=False):
    keys = list(defns.keys())
    if len(defns) == 1:
        return unclusterable_default(keys, return_centers=return_centers)
    dictionary = Dictionary(defns.values())
    if len(dictionary) == 0:
        return unclusterable_default(keys, return_centers=return_centers)
    bow_corpus = [dictionary.doc2bow(document) for document in defns.values()]

    similarity_matrix = vecs.get_en().similarity_matrix(dictionary)
    index = SoftCosineSimilarity(bow_corpus, similarity_matrix, num_best=10)
    affinities = np.zeros((len(defns), len(defns)))

    for row, similarities in enumerate(index):
        for col, similarity in similarities:
            affinities[col, row] = similarity

    if return_centers:
        clust_labels, centers = graph_clust(affinities, return_centers=True)
        return group_clust(keys, clust_labels), [keys[c] for c in centers]
    else:
        clust_labels = graph_clust(affinities)
        return group_clust(keys, clust_labels)


def gloss_graph(lemma_name, pos, return_centers=False):
    session = get_session()
    defns = get_defns(lemma_name, pos, include_wiktionary=True, session=session)
    return softcos(defns, return_centers=return_centers)


class Gloss(SenseClusExp):
    returns_centers = True

    def __init__(self):
        self.clus_func = gloss_graph
        super().__init__(
            ("Gloss",),
            mk_nick("gloss"),
            "Gloss",
            None,
            {},
        )
