from gensim.corpora import Dictionary
from gensim.similarities import SoftCosineSimilarity
from finntk.emb.fasttext import vecs
import numpy as np
from senseclust.utils import graph_clust, group_clust, get_defns
from .base import SenseClusExp
from expcomb.utils import mk_nick
from wikiparse.utils.db import get_session


def softcos(defns):
    dictionary = Dictionary(defns.values())
    bow_corpus = [dictionary.doc2bow(document) for document in defns.values()]

    similarity_matrix = vecs.get_en().similarity_matrix(dictionary)
    index = SoftCosineSimilarity(bow_corpus, similarity_matrix, num_best=10)
    affinities = np.zeros((len(defns), len(defns)))

    for row, similarities in enumerate(index):
        for col, similarity in similarities:
            affinities[col, row] = similarity
    clust_labels = graph_clust(affinities)
    return group_clust(list(defns.keys()), clust_labels)


def gloss_graph(lemma_name, pos):
    session = get_session()
    defns = get_defns(lemma_name, pos, include_wiktionary=True, session=session)
    clus = softcos(defns)
    return clus


class Gloss(SenseClusExp):
    def __init__(self):
        self.clus_func = gloss_graph
        super().__init__(
            ("Gloss",),
            mk_nick("gloss"),
            "Gloss",
            None,
            {},
        )
