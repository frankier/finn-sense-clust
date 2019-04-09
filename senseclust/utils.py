from sklearn.cluster import AffinityPropagation
from sklearn.feature_extraction.text import CountVectorizer
import numpy as np


def group_by(it):
    clus = {}
    for k, v in it:
        clus.setdefault(k, []).append(v)
    return clus


def mat_of_sets(sets):
    vectorizer = CountVectorizer(analyzer=lambda x: x)
    return vectorizer.fit_transform(sets)


def self_xtab(li):
    for idx, e1 in enumerate(li):
        for e2 in li[:idx]:
            yield e1, e2


def graph_clust(affinities):
    (n, n) = affinities.shape
    if n == 1:
        return [0]
    mask = np.ones(affinities.shape, dtype=bool)
    np.fill_diagonal(mask, 0)
    if np.all(affinities[mask].flat == affinities[mask].flat[0]):
        if affinities[mask].flat[0] == 0:
            return list(range(n))
        else:
            return [0] * n
    for damping in [0.5, 0.7, 0.9]:
        af = AffinityPropagation(affinity='precomputed', damping=damping).fit(affinities)
        labels = af.labels_
        if not np.any(np.isnan(labels)):
            return labels
    return [0] * n


def group_clust(labels, clust_labels):
    return group_by(((clust_num, labels[idx]) for idx, clust_num in enumerate(clust_labels)))


def split_line(line):
    frame_id, lemma_id = line.strip().split(",", 1)
    lemma, frame_no = frame_id.split(".", 1)
    return lemma, frame_no, lemma_id
