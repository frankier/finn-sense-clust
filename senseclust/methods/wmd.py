from finntk.emb.fasttext import vecs
from senseclust.utils import graph_clust_grouped, get_defns, unclusterable_default, mk_dictionary_bow_corpus
from .base import SenseClusExp
from expcomb.utils import mk_nick
from wikiparse.utils.db import get_session
from numpy import int, double, zeros, identity, errstate
from itertools import islice
from gensim.corpora.dictionary import Dictionary
from pyemd import emd
from scipy.spatial.distance import cosine
from functools import partial


def mk_distance_matrix(vecs, dictionary):
    assert type(dictionary) is Dictionary  # ids need to be consecutive integers: won't work with HashDictionary
    vocab_len = len(dictionary)
    distance_matrix = zeros((vocab_len, vocab_len), dtype=double)
    for i, t1 in dictionary.items():
        for j, t2 in islice(dictionary.items(), i):
            # Compute Euclidean distance between word vectors.
            distance_matrix[i, j] = distance_matrix[j, i] = cosine(vecs[t1], vecs[t2])
    return distance_matrix


def wmdistance(i, j, mat, norm_mat, lengths, distance_matrix):
    return emd(norm_mat[i], norm_mat[j], distance_matrix)


def wmdistance_partial(i, j, mat, norm_mat, lengths, distance_matrix):
    if lengths[i] < lengths[j]:
        bow_a = norm_mat[i]
        bow_b = mat[j] / lengths[i]
    else:
        bow_a = mat[i] / lengths[j]
        bow_b = norm_mat[j]
    return emd(bow_a, bow_b, distance_matrix, extra_mass_penalty=0)


def mat_of_nbows(dictionary, nbows):
    mat = zeros((len(nbows), len(dictionary)), dtype=double)
    for nbow_idx, nbow in enumerate(nbows):
        for word_id, freq in nbow:
            mat[nbow_idx, word_id] = freq
    return mat


def wmd(wmd_pair, defns, return_centers=False):
    keys = list(defns.keys())
    num_defns = len(defns)
    if num_defns == 1:
        return unclusterable_default(keys, return_centers=return_centers)

    kv = vecs.get_en()
    lengths = zeros(num_defns, dtype=int)

    def gen():
        for defn_idx, defn in enumerate(defns.values()):
            filtered = [token for token in defn if token in kv]
            lengths[defn_idx] = len(filtered)
            yield filtered
    defns_filtered = gen()
    dictionary, bow_corpus = mk_dictionary_bow_corpus(defns_filtered)
    if len(dictionary) == 0:
        return unclusterable_default(keys, return_centers=return_centers)
    distance_matrix = mk_distance_matrix(kv, dictionary)
    mat = mat_of_nbows(dictionary, bow_corpus)

    # div by 0 is okay since we guard against using these values later
    with errstate(divide='ignore'):
        norm_mat = mat / lengths[:, None]

    affinities = identity(num_defns)
    for i in range(num_defns):
        for j in range(i):
            if lengths[i] == 0 or lengths[j] == 0:
                continue
            affinities[i, j] = 1 - wmd_pair(i, j, mat, norm_mat, lengths, distance_matrix)

    return graph_clust_grouped(affinities, keys, return_centers)


def wmd_graph(wmd_pair, lemma_name, pos, return_centers=False):
    session = get_session()
    defns = get_defns(lemma_name, pos, include_wiktionary=True, session=session)
    return wmd(wmd_pair, defns, return_centers=return_centers)


class Wmd(SenseClusExp):
    returns_centers = True

    def __init__(self, partial_match=False):
        self.clus_func = partial(wmd_graph, wmdistance_partial if partial_match else wmdistance)
        super().__init__(
            ("Wmd",),
            mk_nick("wmd", partial_match and "partial" or None),
            "Wmd" + ("Part" if partial_match else ""),
            None,
            {"partial_match": partial_match},
        )