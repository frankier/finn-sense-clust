import numpy as np
from finntk.wordnet.reader import fiwn
from finntk.wsd.lesk_pp import fiwn_vecs
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


def get_num_lemmas():
    return count(lemmas())


def get_vector_dim():
    vector = vecs[next(lemmas())]
    return vector.shape[0]


def mat_of_entities(entities, dim, lim):
    mat = np.zeros((lim, dim))

    for idx, lemma in enumerate(lemmas()):
        if idx >= lim:
            break
        mat[idx, :] = vecs[lemma]
    return mat


def clust_all(lim=100):
    all_vecs = mat_of_entities(lemmas(), get_vector_dim(), lim)

    clust = linkage(all_vecs, metric='cosine')

    print(clust)


def clust_lemma(lemma_name):
    from finntk.wordnet.utils import ss2pre

    synset_ids = set()
    for lemma in fiwn.lemmas(lemma_name):
        synset_ids.add(ss2pre(lemma.synset()))
    print('synset_ids', synset_ids)

    def entities():
        for entity in lemmas():
            print('entity', entity)
            synset_id = entity.split(LEMMA_SEP)[1]
            print(synset_id)
            if synset_id not in synset_ids:
                continue
            yield entity
    entity_list = list(entities())
    if len(entity_list) == 0:
        return
    print('entity_list', entity_list)
    lemma_mat = mat_of_entities(entity_list, get_vector_dim(), len(entity_list))
    print('lemma_mat', lemma_mat)
    Z = linkage(lemma_mat, metric='cosine')
    return hierarchy.dendrogram(Z, labels=[e.split(LEMMA_SEP)[0] for e in entity_list])
