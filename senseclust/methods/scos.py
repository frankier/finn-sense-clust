from gensim.similarities import SoftCosineSimilarity
from gensim.models.keyedvectors import SparseTermSimilarityMatrix, WordEmbeddingSimilarityIndex
from finntk.emb.fasttext import vecs
import numpy as np
from senseclust.utils.clust import graph_clust_grouped, get_defns, unclusterable_default, mk_dictionary_bow_corpus
from .base import SenseClusExp
from expcomb.utils import mk_nick
from functools import partial


def softcos(defns, return_centers=False):
    keys = list(defns.keys())
    if len(defns) == 1:
        return unclusterable_default(keys, return_centers=return_centers)
    dictionary, bow_corpus = mk_dictionary_bow_corpus(defns.values())
    if len(dictionary) == 0:
        return unclusterable_default(keys, return_centers=return_centers)

    similarity_index = WordEmbeddingSimilarityIndex(vecs.get_en())
    similarity_matrix = SparseTermSimilarityMatrix(similarity_index, dictionary)
    index = SoftCosineSimilarity(bow_corpus, similarity_matrix)
    affinities = np.zeros((len(defns), len(defns)))

    for row, similarities in enumerate(index):
        affinities[row] = similarities

    return graph_clust_grouped(affinities, keys, return_centers)


def scos_graph(include_enss, lemma_name, pos, return_centers=False):
    defns = get_defns(lemma_name, pos, lower=True, include_enss=include_enss)
    return softcos(defns, return_centers=return_centers)


class SoftCos(SenseClusExp):
    returns_centers = True

    def __init__(self, include_enss=False):
        self.clus_func = partial(scos_graph, include_enss)
        super().__init__(
            ("SoftCos",),
            mk_nick("scos", include_enss and "enss" or None),
            "SoftCos" + ("+Synset" if include_enss else ""),
            None,
            {"include_enss": include_enss},
        )
