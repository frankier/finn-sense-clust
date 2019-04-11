from gensim.corpora import Dictionary
from gensim.similarities import SoftCosineSimilarity
from finntk.emb.fasttext import vecs
import numpy as np
from senseclust.utils import graph_clust, group_clust, get_defns


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


def gloss_graph(lemma_name, include_wiktionary=False, session=None):
    defns = get_defns(lemma_name, include_wiktionary=include_wiktionary, session=session)
    clus = softcos(defns)
    return clus
