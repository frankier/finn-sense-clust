from gensim.similarities import SoftCosineSimilarity
from gensim.models.keyedvectors import SparseTermSimilarityMatrix, WordEmbeddingSimilarityIndex
from gensim.matutils import corpus2csc
from gensim.utils import is_corpus
from scipy import sparse
from finntk.emb.fasttext import vecs
import numpy as np
from senseclust.utils import graph_clust_grouped, get_defns, unclusterable_default, mk_dictionary_bow_corpus
from .base import SenseClusExp
from expcomb.utils import mk_nick
from functools import partial


class MinLengthSTSM(SparseTermSimilarityMatrix):
    def inner_product(self, X, Y, normalized=True):
        assert normalized

        if not X or not Y:
            return self.matrix.dtype.type(0.0)

        is_corpus_X, X = is_corpus(X)
        is_corpus_Y, Y = is_corpus(Y)

        assert is_corpus_X and is_corpus_Y

        dtype = self.matrix.dtype
        X = corpus2csc(X if is_corpus_X else [X], num_terms=self.matrix.shape[0], dtype=dtype)
        Y = corpus2csc(Y if is_corpus_Y else [Y], num_terms=self.matrix.shape[0], dtype=dtype)
        matrix = self.matrix

        # use the following equality: np.diag(A.T.dot(B).dot(A)) == A.T.dot(B).multiply(A.T).sum(axis=1).T
        X_norm = X.T.dot(matrix).multiply(X.T).sum(axis=1).T
        Y_norm = Y.T.dot(matrix).multiply(Y.T).sum(axis=1).T

        assert \
            X_norm.min() > 0.0 and Y_norm.min() >= 0.0, \
            u"sparse documents must not contain any explicit zero entries and the similarity matrix S " \
            u"must satisfy x^T * S * x > 0 for any nonzero bag-of-words vector x."

        X_norm_sqrt = np.sqrt(X_norm)
        Y_norm_sqrt = np.sqrt(Y_norm)
        min_X_Y_norm_sqrt = np.minimum(X_norm_sqrt, Y_norm_sqrt)
        X = X.multiply(sparse.csr_matrix(1 / min_X_Y_norm_sqrt))
        Y = Y.multiply(sparse.csr_matrix(1 / min_X_Y_norm_sqrt))
        Y[Y == np.inf] = 0  # Account for division by zero when Y_norm.min() == 0.0

        result = X.T.dot(matrix).dot(Y)

        result.data = np.clip(result.data, -1.0, 1.0)

        return result


def softcos(stsm_cls, defns, return_centers=False):
    keys = list(defns.keys())
    if len(defns) == 1:
        return unclusterable_default(keys, return_centers=return_centers)
    dictionary, bow_corpus = mk_dictionary_bow_corpus(defns.values())
    if len(dictionary) == 0:
        return unclusterable_default(keys, return_centers=return_centers)

    similarity_index = WordEmbeddingSimilarityIndex(vecs.get_en())
    similarity_matrix = stsm_cls(similarity_index, dictionary)
    index = SoftCosineSimilarity(bow_corpus, similarity_matrix)
    affinities = np.zeros((len(defns), len(defns)))

    for row, similarities in enumerate(index):
        affinities[row] = similarities

    return graph_clust_grouped(affinities, keys, return_centers)


def scos_graph(stsm_cls, lemma_name, pos, return_centers=False):
    defns = get_defns(lemma_name, pos)
    return softcos(stsm_cls, defns, return_centers=return_centers)


class SoftCos(SenseClusExp):
    returns_centers = True

    def __init__(self, partial_match=False):
        self.clus_func = partial(scos_graph, MinLengthSTSM if partial_match else SparseTermSimilarityMatrix)
        super().__init__(
            ("SoftCos",),
            mk_nick("scos", partial_match and "partial" or None),
            "SoftCos" + ("Part" if partial_match else ""),
            None,
            {"partial_match": partial_match},
        )
